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
    args = parser_args("--checkpt_path",
                        default="./model.pt",
                        type=str,
                        help="Saving dir of the final checkpoint.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_loader, val_loader, test_loader = get_dataloader(args.data_dir, tokenizer)
    train(args.model_name,
          train_loader,
          val_loader,
          args.device,
          args.ngram_size,
          args.n_epochs,
          args.attention,
          args.n_gpu,
          args.checkpt_path)
