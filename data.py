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

class mimic3_dataset(Dataset):

    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.idx = list(range(len(labels)))
        self.labels = labels
        self.tokenizer = tokenizer
        self.ngram_size = ngram_size
        assert (len(self.texts) == len(self.labels))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, key):
        return [self.texts[key], self.idx[key], self.labels[key]]

    def mimic3_col_func(self, batch):
        batch_inputs = self.tokenizer([elem[0] for elem in batch], return_tensors="pt", padding=True)
        input_ids = batch_inputs["input_ids"]
        ngram_encoding = get_ngram_encoding(attn_mask=batch_inputs['attention_mask'],
                                            ngram_size=self.ngram_size,
                                            sep_cls=True)
        labels = torch.cat([elem[2].unsqueeze(0) for elem in batch], dim=0).type('torch.FloatTensor')

        return (input_ids, ngram_encoding, labels)

def get_dataloader(path, tokenizer):
    with open(join(path,'TOP_50_CODES.csv'),'r') as f:
        idx2code = [elem[:-1] for elem in f.readlines()]
        f.close()
    code2idx = {elem:i for i, elem in enumerate(idx2code)}

    train_df = pd.read_csv(join(path,'train_50.csv'))
    val_df = pd.read_csv(join(path,'dev_50.csv'))
    test_df = pd.read_csv(join(path,'test_50.csv'))

    train_texts = [elem[6:-6] for elem in train_df['TEXT']]
    val_texts = [elem[6:-6] for elem in val_df['TEXT']]
    test_texts = [elem[6:-6] for elem in test_df['TEXT']]

    train_codes = [[code2idx[code] for code in elem.split(';')] for elem in train_df['LABELS']]
    val_codes = [[code2idx[code] for code in elem.split(';')] for elem in val_df['LABELS']]
    test_codes = [[code2idx[code] for code in elem.split(';')] for elem in test_df['LABELS']]

    train_labels = [sum([torch.arange(50) == torch.Tensor([code]) for code in sample]) for sample in train_codes]
    val_labels = [sum([torch.arange(50) == torch.Tensor([code]) for code in sample]) for sample in val_codes]
    test_labels = [sum([torch.arange(50) == torch.Tensor([code]) for code in sample]) for sample in test_codes]

    train_dataset = mimic3_dataset(train_texts, train_labels, tokenizer)
    val_dataset = mimic3_dataset(val_texts, val_labels, tokenizer)
    test_dataset = mimic3_dataset(test_texts, test_labels, tokenizer)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size_train,
                              collate_fn=train_dataset.mimic3_col_func,
                              shuffle=True)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size_dev,
                            collate_fn=val_dataset.mimic3_col_func,
                            shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size_test,
                             collate_fn=test_dataset.mimic3_col_func,
                             shuffle=True)

    return train_loader, val_loader, test_loader
