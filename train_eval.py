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
        metrics = all_metrics(yhat, y, k=k, yhat_raw=yhat_raw)
        metrics = all_metrics(yhat, y, k=k, yhat_raw=yhat_raw)
        print_metrics(metrics)

    print('Total eval loss after epoch is {}.'.format(str(total_loss / num_examples)))

def train(model_name, train_loader, val_loader, tokenizer, device, ngram_size, n_epochs, attn, lr, eps, n_gpu, checkpt_path):

    # Define model, parallel training, optimizer.
    if attn:
        model = NGramTransformer_Attn(model_name, ngram_size).to(device)
    else:
        model = NGramTransformer(model_name, ngram_size).to(device)

    if n_gpu > 1:
        device_ids = [_ for _ in range(n_gpu)]
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=float(lr), eps=float(eps))
    criterion = torch.nn.BCEWithLogitsLoss()
    model.zero_grad()

    #Train
    for i in range(n_epochs):
        total_loss = 0.
        num_examples = 0
        for idx, (input_ids, ngram_encoding, labels) in enumerate(train_loader):

            input_ids = input_ids.to(device)
            ngram_encoding = ngram_encoding.to(device)
            labels = labels.to(device)

            model.train()
            logits = model(input_ids, ngram_encoding)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            model.zero_grad()
            total_loss += loss.item() * logits.size()[0]
            num_examples += logits.size()[0]

        print('Average train loss after epoch {} is {}.'.format(str(i+1),str(total_loss / num_examples)))
        eval(model, tokenizer, val_loader, device, ngram_size)
        torch.save(model, checkpt_path)
