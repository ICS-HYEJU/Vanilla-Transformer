import torch

def build_scheduler(cfg, optimizer):
    if cfg['scheduler']['name'] == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=cfg['scheduler']['T_max'],
                                                               eta_min=cfg['scheduler']['eta_min'])
    else:
        raise NotImplementedError('{} is not Implemeted'.format(cfg['scheduler']['name']))