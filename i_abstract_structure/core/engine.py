# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
import time
import os
import numpy as np
#
FP16 = False
from torch.cuda import amp
#
from i_abstract_structure.dataset import train_dataloader
#

class Trainer():
    def __init__(self, cfg, device=torch.device('cpu')):
        self.cfg=cfg
        self.device=device

        # ===== Save Path =====
        self.save_path = self.make_save_path()

        # ===== Tensorboard =====
        # self.tblogger = SummaryWriter(self.save_path)

        # ===== DataLoader =====
        self.train_loader= self.get_dataloader()

        # ===== Model =====
        self.model = self.build_model()

        # ===== Optimizer =====
        self.optimizer = self.build_optimizer()

        # ===== Scheduler =====
        self.scheduler = self.build_scheduler()

        # ===== Loss =====
        self.criterion = self.set_criterion()

        # ===== Parameters =====
        self.max_epoch = self.cfg['solver']['max_epoch']
        self.max_stepnum = len(self.train_loader)

    def calc_loss(self, logits:torch.tensor, targets:torch.tensor):
        logits = logits.view(-1,self.cfg['dataset_info']['vocab_size'])
        targets = targets.view(-1)
        return self.criterion(logits, targets)

    def set_criterion(self):
        return torch.nn.CrossEntropyLoss(ignore_index=self.cfg['dataset_info']['PAD_IDX'])
        # Multi Classification-> CrossEntropy

    def build_scheduler(self):
        from i_abstract_structure.solver.fn_scheduler import build_scheduler
        return build_scheduler(self.cfg, self.optimizer)

    def build_optimizer(self):
        from i_abstract_structure.solver.fn_optimizer import build_optimizer
        return build_optimizer(self.cfg, self.model)

    def build_model(self):
        name = self.cfg['model']['name']
        if name == 'Transformer':
            from i_abstract_structure.model.Transformer import TransformerModel
            model = TransformerModel(seq_len=self.cfg['dataset_info']['seq_len'],
                                     vocab_size=self.cfg['dataset_info']['vocab_size']).to(self.device)
        else:
            raise NotImplementedError(f'The required model is not implemented yet...')
        return model

    def get_dataloader(self):
        train_loader = train_dataloader(self.cfg, mode='train')
        return train_loader

    def make_save_path(self):
        save_path = os.path.join(self.cfg['path']['save_base_path'],
                                 self.cfg['model']['name'])
        os.makedirs(save_path, exist_ok=True)
        return save_path

    def start_train(self):
        try:
            print(f'Training Start..')
            start_time = time.time()
            for epoch in range(self.max_epoch):
                self.train_one_epoch(epoch)
                #
                if epoch % 1 == 0:
                    torch.save({'model': self.model.state_dict()},
                               os.path.join(self.save_path, f'weight_{epoch}.pth'))
                    torch.save({'model': self.model.state_dict()},
                               os.path.join(self.save_path, f'last_weight.pth'))


            print(f'\nTraining completed in {(time.time() - start_time) / 3600:.3f} hours.')


            self.scheduler.step()
        except Exception as _:
            print('ERROR in training loop or eval/save model.')
            raise

    def train_one_epoch(self, epoch):
        # Set Trian Mode
        self.model.train()

        if(FP16):
            scaler = amp.GradScaler()

        dataset_size = 0
        running_loss = 0
        running_accuracy = 0
        accuracy = 0

        bar = tqdm(enumerate(self.train_loader), total = len(self.train_loader))

        for step, data in bar:
            src = data[0].to(self.device)
            trg_input = data[1].to(self.device)
            trg_output = data[2].to(self.device)

            batch_size = src.shape[0]

            if(FP16):
                with amp.autocast(enabled=True):
                    output, logits = self.model(enc_src=src, dec_src=trg_input)
                    loss = self.calc_loss(logits, trg_output)

                    # Scaled loss
                    # To call Scaled Gradients, backward() scaled loss
                    scaler.scale(loss).backward()
                    #  |-> AssertionError : assert outputs.is_cuda or outputs.device.type == 'xla'
                    # scaler.step() first unscales the gradients of the optimizer's assinged params.
                    # If these gradients do not contain infs of NaNs, optimizer.step() is then called.
                    # Otherwise, optimizer.step() is skipped.
                    scaler.step(self.optimizer)

                    # Updates the scale for next iteration
                    scaler.update()

            else:
                output, logits = self.model(enc_src=src, dec_src=trg_input)
                loss = self.calc_loss(logits, trg_output)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()

            # logits.shape = (bs, seq_len, vocab_size)
            # trg_output.shape = (bs, seq_len)

            # zero the parameter graidents
            self.optimizer.zero_grad()

            # loss.item() is transformation loss to Python Float
            # loss.item() is loss of batch data. So, To calculate sum of loss, * (batch_size)
            running_loss += loss.item() * batch_size
            running_accuracy = np.mean(output.view(-1).detach().cpu().numpy() == trg_output.view(-1).detach().cpu().numpy())

            accuracy += running_accuracy

            dataset_size += batch_size
            epoch_loss = running_loss / dataset_size

            bar.set_postfix(
                Epoch=epoch, Train_loss=epoch_loss, LR=self.optimizer.param_groups[0]["lr"],
                accuracy=accuracy / float(step+1)
            )

            #break

        accuracy /= len(self.train_loader)

        # Change learning rate by Scheduler
        if self.scheduler is not None:
            self.scheduler.step()

        return epoch_loss, accuracy


if __name__ == '__main__':
    from i_abstract_structure.config.config import get_config_dict

    # Get configuration
    cfg = get_config_dict()
    # Get Traininer
    trainer = Trainer(cfg)
    # Start train
    trainer.start_train()
    # Start validation
    trainer.start_valid()
