# coding=utf-8

import os
import glob
import logging
import argparse
from os.path import join

import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils import *

class mimic3_dataset(Dataset):

    def __init__(self, texts, labels, ngram_size, tokenizer, use_ngram = False):
        self.texts = texts
        self.idx = list(range(len(labels)))
        self.labels = labels
        self.tokenizer = tokenizer
        self.ngram_size = ngram_size
        assert (len(self.texts) == len(self.labels))
        inputs = self.tokenizer([elem for elem in texts], padding=True)
        self.input_ids = inputs["input_ids"]
        self.attn_mask = inputs['attention_mask']

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, key):
        return [self.idx[key],
                self.texts[key],
                self.input_ids[key],
                self.labels[key],
                self.attn_mask[key]]

    def mimic3_col_func(self, batch):
        input_ids = torch.LongTensor([elem[2] for elem in batch])
        ngram_encoding = get_ngram_encoding(attn_mask=torch.Tensor([elem[4] for elem in batch]),
                                            ngram_size=self.ngram_size,
                                            sep_cls=True)
        labels = torch.cat([elem[2].unsqueeze(0) for elem in batch], dim=0).type('torch.FloatTensor')

        return (input_ids, ngram_encoding, labels)
