from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import os
import pandas as pd
import sentencepiece as spm
from sklearn.model_selection import train_test_split
import glob


class bible_dataset(Dataset):
    def __init__(self, cfg, seq_len, mode='train'):
        super().__init__()
        #
        self.cfg = cfg
        self.data = self.dataset_to_pandas()
        self.seq_len = seq_len
        #
        self.sp_src, self.sp_trg = self.build_subtokenizer()

        assert self.data is not None, f'Invalid task...'
        #
        if mode == 'train':
            self.src = self.data['en'][:int(len(self.data) * 0.7)]
            self.trg = self.data['ko'][:int(len(self.data) * 0.7)]
        else:
            self.src = self.data['en'][int(len(self.data) * 0.7):]
            self.trg = self.data['ko'][int(len(self.data) * 0.7):]

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src = self.en_encode(self.src[idx])  # Encode Input
        trg_input = self.ko_encode(self.trg[idx])  # Decoder Input
        trg_output = trg_input[1:self.seq_len]  # Decoder Label
        trg_output = np.pad(trg_output, (0, 1), 'constant', constant_values=0)

        return torch.Tensor(src).long(), torch.Tensor(trg_input).long(), torch.Tensor(trg_output).long()

    def dataset_to_pandas(self):
        if self.cfg['dataset_info']['name'] == 'bible':
            pass
        else:
            raise ValueError('Invalid dataset name, currently supported [ bible ]')
        #
        data_path = self.cfg['dataset_info']['path']

        # load en dataset and read
        en_train = open(os.path.join(data_path, 'bible_all_en.txt'))
        en_train_content = en_train.read()
        en_train_list = en_train_content.split('\n')

        # load ko dataset and read
        ko_train = open(os.path.join(data_path, 'bible_all_kr.txt'))
        ko_train_content = ko_train.read()
        ko_train_list = ko_train_content.split('\n')

        # make data to pandas
        data = pd.DataFrame()
        data['en_raw'] = en_train_list
        data['ko_raw'] = ko_train_list
        data = data.reset_index(drop=True)

        # Remove the words such as 'Genesis#.#'
        data['en'] = data['en_raw'].apply(lambda x: x.split(' ')[1:])
        data['en'] = data['en'].apply(lambda x: (' ').join(x))
        data['ko'] = data['ko_raw'].apply(lambda x: x.split(' ')[1:])
        data['ko'] = data['ko'].apply(lambda x: (' ').join(x))

        data = data[['en', 'ko']]

        return data

    def build_subtokenizer(self):
        from i_abstract_structure.model.subtokenizer import load_subtokenizer

        sp_src, sp_trg = load_subtokenizer(corpus_src=self.cfg['subtokenizer']['en_corpus'],
                                           corpus_trg=self.cfg['subtokenizer']['ko_corpus'],
                                           vocab_size=self.cfg['dataset_info']['vocab_size'])
        return sp_src, sp_trg

    def en_encode(self, tmpstr: str) -> np.array:
        tmpstr = np.array(self.sp_src.EncodeAsIds(tmpstr))

        if len(tmpstr) > self.seq_len:
            tmpstr = tmpstr[:self.seq_len]

        else:
            tmpstr = np.pad(tmpstr, (0, self.seq_len - len(tmpstr)),
                            'constant', constant_values=self.sp_src.pad_id())
        return tmpstr

    def ko_encode(self, tmpstr: str) -> np.array:
        tmpstr = np.array(self.sp_trg.EncodeAsIds(tmpstr))
        tmpstr = np.insert(tmpstr, 0, self.sp_trg.bos_id())
        if len(tmpstr) > self.seq_len:
            tmpstr = tmpstr[:self.seq_len - 1]
            tmpstr = np.pad(tmpstr, (0, 1), 'constant', constant_values=self.sp_trg.eos_id())

        else:
            tmpstr = np.pad(tmpstr, (0, cfg['dataset_info']['seq_len'] - len(tmpstr)),
                            'constant', constant_values=self.sp_trg.eos_id())
            tmpstr = np.pad(tmpstr, (0, cfg['dataset_info']['seq_len'] - len(tmpstr)),
                            'constant', constant_values=self.sp_trg.eos_id())

        return tmpstr


if __name__ == '__main__':
    from i_abstract_structure.config.config import get_config_dict

    cfg = get_config_dict()
    #
    bible = bible_dataset(cfg=cfg, seq_len=cfg['dataset_info']['seq_len'])
    #
    bible.__getitem__(10)
    bible_dataloader = torch.utils.data.DataLoader(bible, batch_size=cfg['dataset_info']['batch_size'])
    #
    for i, data in enumerate(bible_dataloader):
        print(data)





