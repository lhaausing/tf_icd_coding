import os
import glob
import pickle
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

def eval(model, tokenizer, val_loader, device, ngram_size):
    model.eval()
    total_loss = 0.
    num_examples = 0
    criterion = torch.nn.BCEWithLogitsLoss()
    k = 5
    y = []
    yhat = []
    yhat_raw = []

    with torch.no_grad():
        for idx, (input_ids, ngram_encoding, labels) in enumerate(val_loader):

            # Assign inputs and labels to device
            input_ids = input_ids.to(device)
            ngram_encoding = ngram_encoding.to(device)
            labels = labels.to(device)

            # Evaluate and get results
            logits = model(input_ids, ngram_encoding)
            loss = criterion(logits, labels)
            total_loss += loss.item() * logits.size()[0]
            num_examples += logits.size()[0]
            y.append(labels.cpu().detach().numpy())
            yhat.append(np.round(torch.sigmoid(logits).cpu().detach().numpy()))
            yhat_raw.append(torch.sigmoid(logits).cpu().detach().numpy())

        # Compute scores with results
        y = np.concatenate(y, axis=0)
        yhat = np.concatenate(yhat, axis=0)
        yhat_raw = np.concatenate(yhat_raw, axis=0)

        pickle.dump(yhat_raw,open('../dev_preds','wb'))

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
    parser.add_argument("--model_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="Model checkpoint directory from transformers library or local dir. Tokenizer uses the same name or dir.")
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
    parser.add_argument("--use_ngram",
                        action="store_true",
                        help="Whether use ngram_embeddings.")
    parser.add_argument("--sep_cls",
                        action="store_true",
                        help="Whether seperate the cls token from convolution/ngram.")
    parser.add_argument("--n_gpu",
                        default=1,
                        type=int,
                        help="Suggested to train on multiple gpus if batch size > 8 and n-gram size < 32.")
    parser.add_argument("--device",
                        default="cuda:0",
                        type=str,
                        help="Normally this doesn't matter.")
    args = parser.parse_args()

    #define tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    #load and transform labels
    with open(join(args.data_dir,'TOP_50_CODES.csv'),'r') as f:
        idx2code = [elem[:-1] for elem in f.readlines()]
        f.close()
    code2idx = {elem:i for i, elem in enumerate(idx2code)}
    #get dataloader
    val_df = pd.read_csv(join(args.data_dir,'dev_50.csv'),engine='python')
    val_texts = [elem[6:-6] for elem in val_df['TEXT']]
    val_codes = [[code2idx[code] for code in elem.split(';')] for elem in val_df['LABELS']]
    val_labels = [sum([torch.arange(50) == torch.Tensor([code]) for code in sample]) for sample in val_codes]
    val_dataset = mimic3_dataset(val_texts, val_labels, args.ngram_size, tokenizer, args.use_ngram, args.sep_cls)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=args.batch_size,
                            collate_fn=val_dataset.mimic3_col_func,
                            shuffle=False)

    model = torch.load(args.model_dir)
    #train
    eval(model, tokenizer, val_loader, device, ngram_size)

if __name__ == '__main__':
    main()
