# coding=utf-8

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

logger = logging.getLogger(__name__)

def eval(args, model, val_loader):
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
            if args.use_ngram:
                logits = model(input_ids, ngram_encoding)
                loss = criterion(logits.to(args.device), labels.to(args.device))
                total_loss += loss.item() * logits.size()[0]
                num_examples += logits.size()[0]
                y.append(labels.cpu().detach().numpy())
                yhat.append(np.round(torch.sigmoid(logits).cpu().detach().numpy()))
                yhat_raw.append(torch.sigmoid(logits).cpu().detach().numpy())

        # Compute scores with results
        y = np.concatenate(y, axis=0)
        yhat = np.concatenate(yhat, axis=0)
        yhat_raw = np.concatenate(yhat_raw, axis=0)
        metrics = all_metrics(yhat, y, k=k, yhat_raw=yhat_raw)
        metrics = all_metrics(yhat, y, k=k, yhat_raw=yhat_raw)

        logger.info('validation loss is %s.', total_loss/num_examples)
        logger.info("[MACRO] acc, prec, rec, f1, auc")
        logger.info("%s, %s, %s, %s, %s", metrics["acc_macro"], metrics["prec_macro"], metrics["rec_macro"], metrics["f1_macro"], metrics["auc_macro"])
        logger.info("[MICRO] accuracy, precision, recall, f-measure, AUC")
        logger.info("%s, %s, %s, %s, %s", metrics["acc_micro"], metrics["prec_micro"], metrics["rec_micro"], metrics["f1_micro"], metrics["auc_micro"])

        for metric, val in metrics.items():
            if metric.find("rec_at") != -1:
                logger.info("%s: %s" % (metric, val))

        return metrics

def train(args, train_loader, val_loader):
    # Define model, parallel training, optimizer.
    if args.use_ngram: model = NGramTransformer(args.model_name,args.ngram_size)
    else: model = local_bert(args.model_name)
    model = model.to(args.device)

    if args.n_gpu > 1:
        device_ids = [_ for _ in range(args.n_gpu)]
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],lr=float(args.lr),eps=float(args.eps))

    criterion = torch.nn.BCEWithLogitsLoss()
    model.zero_grad()

    best_f1 = 0.
    best_auc = 0.

    #Train
    for i in range(args.n_epochs):
        total_loss = 0.
        num_examples = 0
        for idx, (input_ids, attn_mask, labels) in enumerate(train_loader):

            model.train()
            if args.use_ngram:
                ngram_encoding = get_ngram_encoding(attn_mask.to(args.device), args.ngram_size).cpu()
                logits = model(input_ids, ngram_encoding)
            else:
                logits = model(input_ids)
            loss = criterion(logits.to(args.device), labels.to(args.device))

            loss.backward()
            optimizer.step()
            model.zero_grad()
            total_loss += loss.item() * logits.size()[0]
            num_examples += logits.size()[0]

        logger.info('')
        logger.info('epoch: %d', i+1)
        logger.info('train loss is %s.', total_loss / num_examples)

        metrics = eval(model, val_loader)

        if args.save_best_f1:
            if metrics["f1_micro"] > best_f1:
                best_f1 = metrics["f1_micro"]
                torch.save(model, args.checkpt_path+'_best_f1.pt')
        if args.save_best_f1:
            if metrics["auc_micro"] > best_auc:
                best_auc = metrics["auc_micro"]
                torch.save(model, args.checkpt_path+'_best_auc.pt')

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def main():
    parser = argparse.ArgumentParser()

    #required parameters
    parser.add_argument("--model_name", default=None, type=str, required=True,
                        help="Model name or directory from transformers library or local dir. Tokenizer uses the same name or dir.")
    parser.add_argument("--attention", action="store_true",
                        help="Whether plug in the attention layer after the Transformers LM.")
    parser.add_argument("--use_ngram", action="store_true",
                        help="Whether use ngram_embeddings.")
    parser.add_argument("--n_epochs", default=30, type=int,
                        help="Number of epochs of training.")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size for training and validation.")
    parser.add_argument("--ngram_size", default=32, type=int,
                        help="Size of the N-Gram that one's using.")
    parser.add_argument("--maxpool_size", default=32, type=int,
                        help="Size of the Max-pooling. Probably need to be larger than 28.")
    parser.add_argument("--n_gpu", default=1, type=int,
                        help="Suggested to train on multiple gpus if batch size > 8 and n-gram size < 32.")
    parser.add_argument("--lr", default="2e-5", type=str,
                        help="Learning Rate.")
    parser.add_argument("--eps", default="1e-8", type=str,
                        help="Epsilon.")
    parser.add_argument("--device", default="cuda:0", type=str,
                        help="Normally this doesn't matter.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. It should contain a training set and a validation set.")
    parser.add_argument("--load_data_cache", action="store_true",
                        help="load_data_cache.")
    parser.add_argument("--checkpt_path", default="./model", type=str,
                        help="Saving dir of the final checkpoint.")
    parser.add_argument("--save_best_f1", action="store_true",
                        help="save best f1 checkpoints.")
    parser.add_argument("--save_best_auc", action="store_true",
                        help="save best auc checkpoints.")

    args = parser.parse_args()
    set_seed(args)

    #define tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.load_data_cache:
        train_dataset = pickle.load(open(join(args.data_dir,'train_50.pkl'),'rb'))
        val_dataset = pickle.load(open(join(args.data_dir,'dev_50.pkl'),'rb'))
        test_dataset = pickle.load(open(join(args.data_dir,'test_50.pkl'),'rb'))

    else:
        #load csv file
        train_df = pd.read_csv(join(args.data_dir,'train_50.csv'),engine='python')
        val_df = pd.read_csv(join(args.data_dir,'dev_50.csv'),engine='python')
        test_df = pd.read_csv(join(args.data_dir,'test_50.csv'),engine='python')

        #load text
        train_texts = [elem[6:-6] for elem in train_df['TEXT']]
        val_texts = [elem[6:-6] for elem in val_df['TEXT']]
        test_texts = [elem[6:-6] for elem in test_df['TEXT']]

        #load and transform labels
        with open(join(args.data_dir,'TOP_50_CODES.csv'),'r') as f:
            idx2code = [elem[:-1] for elem in f.readlines()]
            f.close()
        code2idx = {elem:i for i, elem in enumerate(idx2code)}

        train_codes = [[code2idx[code] for code in elem.split(';')] for elem in train_df['LABELS']]
        val_codes = [[code2idx[code] for code in elem.split(';')] for elem in val_df['LABELS']]
        test_codes = [[code2idx[code] for code in elem.split(';')] for elem in test_df['LABELS']]

        train_labels = [sum([torch.arange(50) == torch.Tensor([code]) for code in sample]) for sample in train_codes]
        val_labels = [sum([torch.arange(50) == torch.Tensor([code]) for code in sample]) for sample in val_codes]
        test_labels = [sum([torch.arange(50) == torch.Tensor([code]) for code in sample]) for sample in test_codes]

        #build dataset and dataloader
        train_dataset = mimic3_dataset(train_texts, train_labels, args.ngram_size, tokenizer, args.use_ngram)
        val_dataset = mimic3_dataset(val_texts, val_labels, args.ngram_size, tokenizer, args.use_ngram)
        test_dataset = mimic3_dataset(test_texts, test_labels, args.ngram_size, tokenizer, args.use_ngram)

        pickle.dump(train_dataset, open(join(args.data_dir,'train_50.pkl'),'wb'))
        pickle.dump(val_dataset, open(join(args.data_dir,'dev_50.pkl'),'wb'))
        pickle.dump(test_dataset, open(join(args.data_dir,'test_50.pkl'),'wb'))

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              collate_fn=train_dataset.mimic3_col_func,
                              shuffle=True)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=args.batch_size,
                            collate_fn=val_dataset.mimic3_col_func,
                            shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             collate_fn=test_dataset.mimic3_col_func,
                             shuffle=True)

    #train
    train(args, train_loader, val_loader)

if __name__ == '__main__':
    main()
