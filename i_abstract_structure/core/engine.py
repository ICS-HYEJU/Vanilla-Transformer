import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pprint import pprint
from nltk.tokenize import word_tokenize as en_tokenizer
import urllib.request
import csv
import numpy as np
from torch.cuda import amp
from tqdm import tqdm
import time
import copy
from collections import defaultdict
from sklearn.metrics import mean_squared_error
import joblib
import gc
import os
from sklearn.model_selection import train_test_split
import os


#
# from ..utils.events import write_tbimg, write_tbloss, write_tbacc, write_tbPR
# from ..dataset import create_dataloader
#
class Trainer():
    def __init__(self, cfg, device=torch.device("cpu")):
        self.cfg = cfg
        self.device = device

        # ===== save path =====
        self.save_path = self.make_save_path()

        # ===== DataLoader =====
        self.train_loader, self.val_loader = self.get_dataloader()

        # ===== subtokenizer =====
        # self.sp_src, self.sp_trg = self.build_subtokenizer()

        # ===== Optimizer =====
        self.optimizer = self.build_optimizer()

        # ===== Scheduler =====
        self.scheduler = self.build_scheduler()

        # ===== Loss =====
        self.criterion = self.set_criterion()

        # ===== Parameters =====
        self.max_epoch = self.cfg['solver']['max_epoch']
        self.max_stepnum = len(self.train_loader)

    # def build_subtokenizer(self):
    #     from i_abstract_structure.model.subtokenizer import load_subtokenizer
    #
    #     sp_src, sp_trg = load_subtokenizer(corpus_src=self.cfg['subtokenizer']['en_corpus'],
    #                                        corpus_trg=self.cfg['subtokenizer']['ko_corpus'],
    #                                        vocab_size=self.cfg['dataset_info']['vocab_size'])
    #     return sp_src, sp_trg

    def calc_loss(self, logits, labels):
        return self.criterion(logits, labels.float())

    def set_criterion(self):
        return torch.nn.CrossEntropyLoss(ignore_index=self.cfg['dataset_info']['PAD_IDX']).to(self.device)

   #  def build_scheduler(self):
   #      from solver.fn_scheduler import build_scheduler
   #      return build_scheduler(self.cfg, self.optimizer)
   #
   #  def build_optimizer(self):
   #      from solver.fn_optimizer import build_optimizer
   #      optim = build_optimizer(self.cfg, self.model)
   #      return optim
   #
   # def build_model(self):

   # def get_dataloader(self):
   #     train_loader, val_loader = create_dataloader(cfg['dataset_info'])


if __name__ == '__main__':
    from i_abstract_structure.config.config import get_config_dict

    cfg = get_config_dict()

    trainer = Trainer(cfg)
