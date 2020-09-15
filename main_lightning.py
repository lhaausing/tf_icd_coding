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

import pytorch_lightning as pl
from pytorch_lightning import Trainer

import transformers
from transformers import AutoTokenizer, AutoModel

from utils import all_metrics, print_metrics

model_name = 'bert-base-uncased'
device = 'cuda:0'
num_epochs = 50
ngram_size = 32
batch_size_train = 32
batch_size_dev = 32
batch_size_test = 32
path = '/scratch/xl3119/Multi-Filter-Residual-Convolutional-Neural-Network/data/mimic3'
use_attention = False
multi_gpu = True
num_gpu = 3

def get_ngram_encoding(attn_mask = None, ngram_size = None, sep_cls = True):

    sent_lens = torch.sum(attn_mask,1)
    if sep_cls:
        sent_lens -= 1
    max_sent_len = torch.max(sent_lens).item()
    num_ngram = [math.ceil(elem / ngram_size) for elem in sent_lens.tolist()]
    max_num_ngram = max(num_ngram)
    arange_t = torch.arange(max_sent_len)

    ngram_pos = [[min(j * ngram_size, sent_lens[i].item()) for j in range(elem+1)] for i, elem in enumerate(num_ngram)]
    for i in range(len(ngram_pos)):
        ngram_pos[i] = ngram_pos[i] + [-1]*(max_num_ngram+1-len(ngram_pos[i]))
    ngram_encoding = [torch.cat([((arange_t>=elem[i])*(arange_t<elem[i+1])).unsqueeze(0) for i in range(max_num_ngram)]).unsqueeze(0) for elem in ngram_pos]
    ngram_encoding = torch.cat(ngram_encoding)

    if sep_cls:
        size = ngram_encoding.size()
        zero_pos = torch.zeros(size[0],size[1],1,dtype=torch.bool)
        cls_pos = torch.BoolTensor([[[1]+[0]*(size[2])]]*size[0])
        ngram_encoding = torch.cat([zero_pos, ngram_encoding], dim=2)
        ngram_encoding = torch.cat([cls_pos, ngram_encoding], dim=1)

    return ngram_encoding.type(torch.FloatTensor)

class mimic3_dataset(Dataset):

    def __init__(self, texts, labels, tokenizer):

        self.texts = texts
        self.idx = list(range(len(labels)))
        self.labels = labels
        self.tokenizer = tokenizer ##defined in first lines
        self.ngram_size = ngram_size ##defined in first lines
        assert (len(self.texts) == len(self.labels))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, key):

        return [self.texts[key], self.idx[key], self.labels[key]]

    def mimic3_col_func(self, batch):

        batch_inputs = tokenizer([elem[0] for elem in batch], return_tensors="pt", padding=True)

        input_ids = batch_inputs["input_ids"]
        attn_mask = batch_inputs['attention_mask']

        ngram_encoding = get_ngram_encoding(attn_mask=attn_mask, ngram_size=self.ngram_size, sep_cls=True)

        labels = torch.cat([elem[2] for elem in batch], dim = 0).type('torch.FloatTensor')

        return (input_ids, ngram_encoding, labels)

class NGramTF(pl.LightningModule):

    def __init__(self, model_name='', ngram_size = 32, n_class = 50):
        super().__init__()
        if not model_name:
            raise NameError('You have to give a model name from transformers library.')

        self.model_name = model_name
        self.n_class = n_class
        self.ngram_size = ngram_size

        ##Model Layers
        self.bert = AutoModel.from_pretrained(self.model_name)
        self.wd_emb = self.bert.embeddings.word_embeddings
        self.out_layer = nn.Linear(self.bert.config.hidden_size, self.n_class)

    def forward(self, input_ids=None, ngram_encoding=None):
        embeds = torch.bmm(ngram_encoding, self.wd_emb(input_ids))
        embeds, cls_embeds  = self.bert(inputs_embeds=embeds)
        logits = self.out_layer(embeds[:,0,:])

        return logits

    def training_step(self, batch, batch_nb):
        # batch
        input_ids, attn_mask, labels = batch

        # fwd
        logits = self(input_ids, attn_mask)

        # loss
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        # logs
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # batch
        input_ids, attn_mask, labels = batch

        # fwd
        logits = self(input_ids, attn_mask)

        # loss
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-05, eps=1e-08)

    def train_dataloader(self):
        return top50_train_loader

    def val_dataloader(self):
        return top50_val_loader

    def test_dataloader(self):
        return top50_test_loader
