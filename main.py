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

import transformers
from transformers import AutoTokenizer, AutoModel

from data import *
from utils import *
from models import *
from train_eval import *

def main():
    parser = argparse.ArgumentParser()

    #required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. It should contain a training set and a validation set.")
    parser.add_argument("--model_name",
                        default=None,
                        type=str,
                        required=True,
                        help="Model name or directory from transformers library or local dir. Tokenizer uses the same name or dir.")
    parser.add_argument("--n_epochs",
                        default=30,
                        type=int,
                        help="Number of epochs of training.")
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Batch size for training and validation.")
    parser.add_argument("--ngram_size",
                        default=32,
                        type=int,
                        help="Size of the N-Gram that one's using.")
    parser.add_argument("--attention",
                        action="store_true",
                        help="Whether plug in the attention layer after the Transformers LM.")
    parser.add_argument("--n_gpu",
                        default=1,
                        type=int,
                        help="Suggested to train on multiple gpus if batch size > 8 and n-gram size < 32.")
    parser.add_argument("--device",
                        default="cuda:0",
                        type=str,
                        help="Normally this doesn't matter.")
    parser.add_argument("--checkpt_path",
                        default="./model.pt",
                        type=str,
                        help="Saving dir of the final checkpoint.")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    with open(join(args.data_dir,'TOP_50_CODES.csv'),'r') as f:
        idx2code = [elem[:-1] for elem in f.readlines()]
        f.close()
    code2idx = {elem:i for i, elem in enumerate(idx2code)}

    train_df = pd.read_csv(join(args.data_dir,'train_50.csv'),engine='python')
    val_df = pd.read_csv(join(args.data_dir,'dev_50.csv'),engine='python')
    test_df = pd.read_csv(join(args.data_dir,'test_50.csv'),engine='python')

    train_texts = [elem[6:-6] for elem in train_df['TEXT']]
    val_texts = [elem[6:-6] for elem in val_df['TEXT']]
    test_texts = [elem[6:-6] for elem in test_df['TEXT']]

    train_codes = [[code2idx[code] for code in elem.split(';')] for elem in train_df['LABELS']]
    val_codes = [[code2idx[code] for code in elem.split(';')] for elem in val_df['LABELS']]
    test_codes = [[code2idx[code] for code in elem.split(';')] for elem in test_df['LABELS']]

    train_labels = [sum([torch.arange(50) == torch.Tensor([code]) for code in sample]) for sample in train_codes]
    val_labels = [sum([torch.arange(50) == torch.Tensor([code]) for code in sample]) for sample in val_codes]
    test_labels = [sum([torch.arange(50) == torch.Tensor([code]) for code in sample]) for sample in test_codes]

    train_dataset = mimic3_dataset(train_texts, train_labels, args.ngram_size, tokenizer)
    val_dataset = mimic3_dataset(val_texts, val_labels, args.ngram_size, tokenizer)
    test_dataset = mimic3_dataset(test_texts, test_labels, args.ngram_size, tokenizer)

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

    train(args.model_name,
          train_loader,
          val_loader,
          args.device,
          args.ngram_size,
          args.n_epochs,
          args.attention,
          args.n_gpu,
          args.checkpt_path)

if __name__ == '__main__':
    main()
