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
        self.use_ngram = use_ngram
        assert (len(self.texts) == len(self.labels))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, key):
        return [self.texts[key], self.idx[key], self.labels[key]]

    def mimic3_col_func(self, batch):
        batch_inputs = self.tokenizer([elem[0] for elem in batch], return_tensors="pt", padding=True)
        input_ids = batch_inputs["input_ids"]
        if self.use_ngram:
            ngram_encoding = get_ngram_encoding(attn_mask=batch_inputs['attention_mask'],
                                                ngram_size=self.ngram_size,
                                                sep_cls=True)
        else:
            ngram_encoding = None
        labels = torch.cat([elem[2].unsqueeze(0) for elem in batch], dim=0).type('torch.FloatTensor')

        return (input_ids, ngram_encoding, labels)
