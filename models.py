import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import AutoTokenizer, AutoModel

from utils import all_metrics, print_metrics

class Attn_Layer(nn.Module):
    """
    Calculate attention for each label.
    After transposition, attn.size() = (batch_size, class_size, max_sent_len)
    """

    def __init__(self, hid, class_size):
        super().__init__()
        self.hid = hid
        self.class_size = class_size
        self.w = nn.Linear(self.hid, self.class_size)

    def forward(self, input_embeds):
        attn_w = self.w(input_embeds)
        attn_w = torch.transpose(attn_w, 1, 2)
        attn_w = F.softmax(attn_weights, dim=2)

        embeds = torch.bmm(attn_w, input_embeds)

        return embeds

class Attn_Out_Layer(nn.Module):
    """Calculate logits before applying sigmoid func."""

    def __init__(self, hid, class_size):
        super().__init__()
        self.hid = hid
        self.class_size = class_size
        self.out_w = nn.Linear(self.hid, 1)

    def forward(self, input_embeds):
        logits = self.out_w(input_embeds)
        logits = logit.view(-1, self.class_size)

        return logits

class NGramTransformer(nn.Module):

    def __init__(self, model_name='', ngram_size = 32, n_class = 50):
        super().__init__()
        self.ngram_size = ngram_size
        self.model_name = model_name
        self.bert = AutoModel.from_pretrained(model_name)
        self.hid = self.bert.config.hidden_size
        self.out_layer = nn.Linear(self.hid, n_class)
        self.wd_emb = self.bert.embeddings.word_embeddings

    def forward(self, input_ids=None, ngram_encoding=None):
        embeds = torch.bmm(ngram_encoding, self.wd_emb(input_ids))
        embeds, cls_embeds  = self.bert(inputs_embeds=embeds)
        logits = self.out_layer(embeds[:,0,:])

        return logits

class NGramTransformer_Attn(nn.Module):

    def __init__(self, model_name='', ngram_size = 32, n_class = 50,device= 'cuda:0'):
        super().__init__()
        self.model_name = model_name
        self.bert = AutoModel.from_pretrained(model_name)
        self.hid = self.bert.config.hidden_size
        self.class_size = n_class
        self.ngram_size = ngram_size

        self.wd_emb = self.bert.embeddings.word_embeddings
        self.attn_layer = Attn_Layer(self.hid, self.class_size)
        self.out_layer = Attn_Out_Layer(self.hid, self.class_size)

    def forward(self, input_ids=None, ngram_encoding=None):
        embeds = torch.bmm(ngram_encoding, self.wd_emb(input_ids))
        embeds, cls_embeds  = self.bert(inputs_embeds=embeds)
        embeds = self.attn_layer(embeds)
        logits = self.out_layer(embeds)

        return logits

class cnn_bert(nn.Module):
    def __init__(self, model_name='', ngram_size = 16, mp_size = 32, n_class = 50, device= 'cuda:0', use_attn = False):
        super().__init__()
        #Transformers Encoder
        self.bert = AutoModel.from_pretrained(model_name)

        #some_names
        self.model_name = model_name
        self.hid = self.bert.config.hidden_size
        self.c = n_class
        self.ngram_size = ngram_size
        self.mp_size = mp_size
        self.use_attn = use_attn

        #layers
        self.wd_emb = self.bert.embeddings.word_embeddings
        self.conv = nn.Conv1d(self.hid, self.hid, self.ngram_size)
        self.maxpool = nn.MaxPool1d(self.mp_size)
        self.attn = Attn_Layer(self.hid, self.c)
        self.out = Attn_Out_Layer(self.hid, self.c)

    def forward(self, input_ids=None):
        x = self.wd_emb(input_ids)
        x = self.conv_layer(x.permute(0,2,1))
        x = self.max_pool(x)
        x, x_cls  = self.bert(inputs_embeds=x.permute(0,2,1))
        if self.use_attn:
            x = self.attn(x)
            logits = self.out(x)
        else:
            logits = self.out_layer(x[:,0,:])

        return logits
