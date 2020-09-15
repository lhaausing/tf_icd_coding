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

def calculate_ngram_position_matrix(attn_mask = None, max_ngram_size = None, model_device = None, sep_cls = True):

    for i, elem in enumerate([attn_mask, max_ngram_size, model_device]):
        if elem is None:
            raise NameError('You must give input in position {}'.format(i))

    if attn_mask.device != model_device:
        attn_mask = attn_mask.to(model_device)

    batch_sent_lens = torch.sum(attn_mask,1)
    if sep_cls:
        batch_sent_lens -= 1
    max_sent_len = torch.max(batch_sent_lens).item()
    num_n_grams = [math.ceil(elem / max_ngram_size) for elem in batch_sent_lens.tolist()]
    max_num_n_grams = max(num_n_grams)
    arange_t = torch.arange(max_sent_len)

    n_gram_pos = [[min(j * max_ngram_size, batch_sent_lens[i].item()) for j in range(elem+1)] for i, elem in enumerate(num_n_grams)]
    for i in range(len(n_gram_pos)):
        n_gram_pos[i] = n_gram_pos[i] + [-1]*(max_num_n_grams+1-len(n_gram_pos[i]))
    n_gram_pos_matrix = [torch.cat([((arange_t>=elem[i])*(arange_t<elem[i+1])).unsqueeze(0) for i in range(max_num_n_grams)]).unsqueeze(0) for elem in n_gram_pos]
    n_gram_pos_matrix = torch.cat(n_gram_pos_matrix).to(model_device)

    if sep_cls:
        size = n_gram_pos_matrix.size()
        zero_pos = torch.zeros(size[0],size[1],1,dtype=torch.bool).to(model_device)
        cls_pos = torch.BoolTensor([[[1]+[0]*(size[2])]]*size[0]).to(model_device)
        n_gram_pos_matrix = torch.cat([zero_pos, n_gram_pos_matrix], dim=2)
        n_gram_pos_matrix = torch.cat([cls_pos, n_gram_pos_matrix], dim=1)

    return n_gram_pos_matrix.type(torch.FloatTensor).to(model_device)

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

        output = torch.bmm(attn_w, embeds)

        return output

class Attn_Out_Layer(object):
    """Calculate logits before applying sigmoid func."""

    def __init__(self, hidden_size, class_size):
        super().__init__()
        self.hid_size = hidden_size
        self.class_size = class_size
        self.out_w = nn.Linear(self.hid_size, 1)

    def forward(input_embeds):
        logit = self.out_w(input_embeds)
        logit = logit.view(-1, self.class_size)

        return logit


class NGramTransformer_Attn(nn.Module):

    def __init__(self, model_name='', max_n_gram_len = 32, n_class = 50,device= 'cuda:0'):
        super().__init__()
        if not model_name:
            raise NameError('You have to give a model name from transformers library.')
        self.transformers_model = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.transformers_model.config.hidden_size
        self.class_size = n_class
        self.max_n_gram_len = max_n_gram_len
        self.device = device

        self.word_embeddings = self.transformers_model.embeddings.word_embeddings
        self.attn_layer = nn.Linear(self.hidden_size, self.class_size)
        self.out_layer = nn.Linear(self.hidden_size, 1)


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        embeds = self.word_embeddings(input_ids)
        ngram_pos_matrix = calculate_ngram_position_matrix(attn_mask=attention_mask,
                                                           max_ngram_size=self.max_n_gram_len,
                                                           model_device=self.device)

        embeds = torch.bmm(ngram_pos_matrix, embeds)
        embeds, cls_embeds  = self.transformers_model(inputs_embeds=embeds)
        #logit = self.out_layer(embeds[:,0,:])
        attn_weights = torch.transpose(self.attn_layer(embeds), 1, 2)
        attn_weights = F.softmax(attn_weights)
        attn_outputs = torch.bmm(attn_weights,embeds)
        logit = self.out_layer(attn_outputs)
        logit = logit.view(-1, self.class_size)

class NGramTransformer(nn.Module):

    def __init__(self, model_name='', max_n_gram_len = 32, n_class = 50,device= 'cuda:0'):
        super().__init__()
        if not model_name:
            raise NameError('You have to give a model name from transformers library.')
        self.transformers_model = AutoModel.from_pretrained(model_name)
        self.out_layer = nn.Linear(self.transformers_model.config.hidden_size, n_class)
        self.word_embeddings = self.transformers_model.embeddings.word_embeddings
        self.max_n_gram_len = max_n_gram_len
        self.device = device

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        embeds = self.word_embeddings(input_ids)
        ngram_pos_matrix = calculate_ngram_position_matrix(attn_mask=attention_mask,
                                                           max_ngram_size=self.max_n_gram_len,
                                                           model_device=self.device)

        embeds = torch.bmm(ngram_pos_matrix, embeds)
        embeds, cls_embeds  = self.transformers_model(inputs_embeds=embeds)
        logit = self.out_layer(embeds[:,0,:])

        return logit
