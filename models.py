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

    def __init__(self, hidden_size, class_size):
        super().__init__()
        self.hid_size = hidden_size
        self.class_size = class_size
        self.w = nn.Linear(self.hid_size, self.class_size)

    def forward(self, input_embeds):
        attn_w = self.w(input_embeds)
        attn_w = torch.transpose(attn_w, 1, 2)
        attn_w = F.softmax(attn_weights, dim=2)

        embeds = torch.bmm(attn_w, input_embeds)

        return embeds

class Attn_Out_Layer(nn.Module):
    """Calculate logits before applying sigmoid func."""

    def __init__(self, hidden_size, class_size):
        super().__init__()
        self.hid_size = hidden_size
        self.class_size = class_size
        self.out_w = nn.Linear(self.hid_size, 1)

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
        self.hidden_size = self.bert.config.hidden_size
        self.out_layer = nn.Linear(self.hidden_size, n_class)
        self.wd_emb = self.bert.embeddings.word_embeddings

    def forward(self, input_ids=None, ngram_encoding=None):
        embeds = torch.bmm(ngram_encoding, self.wd_emb(input_ids))
        if "distil" in self.model_name:
            embeds = self.bert(input_embeds=embeds)
        else:
            embeds, cls_embeds  = self.bert(inputs_embeds=embeds)
        logits = self.out_layer(embeds[:,0,:])

        return logits

class NGramTransformer_Attn(nn.Module):

    def __init__(self, model_name='', ngram_size = 32, n_class = 50,device= 'cuda:0'):
        super().__init__()
        self.model_name = model_name
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        self.class_size = n_class
        self.ngram_size = ngram_size

        self.wd_emb = self.bert.embeddings.word_embeddings
        self.attn_layer = Attn_Layer(self.hidden_size, self.class_size)
        self.out_layer = Attn_Out_Layer(self.hidden_size, self.class_size)

    def forward(self, input_ids=None, ngram_encoding=None):
        embeds = torch.bmm(ngram_encoding, self.wd_emb(input_ids))
        if "distil" in self.model_name:
            embeds = self.bert(input_embeds=embeds)
        else:
            embeds, cls_embeds  = self.bert(inputs_embeds=embeds)
        embeds = self.attn_layer(embeds)
        logits = self.out_layer(embeds)

        return logits
