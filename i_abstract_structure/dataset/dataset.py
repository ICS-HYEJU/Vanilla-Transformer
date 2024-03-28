from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import pandas as pd
import sentencepiece as spm
from sklearn.model_selection import train_test_split


class bible_dataset(Dataset):
    def __init__(self, cfg, seq_len):
        super().__init__()
        #
        self.cfg = cfg
        self.data = self.dataset_to_pandas(cfg)
        self.seq_len = seq_len
        #
        self.sp_src, self.sp_trg = self.build_subtokenizer()
        self.src_data = self.data['en']
        self.trg_data = self.data['ko']

        assert self.src_data == True or self.trg_data == True, f'Invalid task...'
        #
        self.src_list = []
        self.trg_list = []
        #

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        self.src_list.append(self.en_encode(self.src_data[idx]))
        self.trg_list.append(self.ko_encode(self.trg_data[idx]))

        return self.src_list, self.trg_list


    def dataset_to_pandas(self, cfg_dataset):
        if cfg_dataset['name'] == 'bible':
            pass
        else:
            raise ValueError('Invalid dataset name, currently supported [ bible ]')
        #
        data_path = cfg_dataset['path']

        # load en dataset and read
        en_train = open(os.path.join(data_path, 'bible-all.en.txt'))
        en_train_content = en_train.read()
        en_train_list = en_train_content.split('\n')

        # load ko dataset and read
        ko_train = open(os.path.join(data_path, 'bible-all.ko.txt'))
        ko_train_content = ko_train.read()
        ko_train_list = ko_train_content.split('\n')

        # make data to pandas
        data = pd.DataFrame()
        data['en_raw'] = en_train_list
        data['ko_raw'] = ko_train_list
        data = data.set_index(drop=True)

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

    def split_dataset(self):
        src_train, src_valid, trg_train, trg_valid = train_test_split(self.src_list, self.trg_list, test_size=0.2,
                                                                      random_state=42)
        return src_train, src_valid, trg_train, trg_valid

if __name__ == '__main__':
    from i_abstract_structure.config.config import get_config_dict
    cfg = get_config_dict()
    #
    bible = bible_dataset(cfg=cfg, seq_len = cfg['dataset_info']['seq_len'])
    #
    src_train, src_valid, trg_train, trg_valid = bible.split_dataset()
    #
    loader =DataLoader(bible, batch_size=cfg['dataset_info']['batch_size'])

