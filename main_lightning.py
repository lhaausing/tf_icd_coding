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
max_n_gram_len = 32
batch_size_train = 8
batch_size_dev = 8
batch_size_test = 8
path = '/scratch/xl3119/Multi-Filter-Residual-Convolutional-Neural-Network/data/mimic3'
use_attention = False
multi_gpu = True
num_gpu = 3

class mimic3_dataset(Dataset):

    def __init__(self, texts, labels):

        self.texts = texts
        self.idx = list(range(len(labels)))
        self.labels = labels
        assert (len(self.texts) == len(self.labels))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, key):

        return [self.texts[key], self.idx[key], self.labels[key]]

class NGramTF(pl.LightningModule):

    def __init__(self, model_name='', max_n_gram_len = 32, n_class = 50):
        super().__init__()
        if not model_name:
            raise NameError('You have to give a model name from transformers library.')

        self.model_name = model_name
        self.n_class = n_class
        self.max_n_gram_len = max_n_gram_len

        ##Model Layers
        self.tokenizer = = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(self.model_name)
        self.word_embeddings = self.bert.embeddings.word_embeddings
        self.out_layer = nn.Linear(self.bert.config.hidden_size, self.n_class)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        embeds = self.word_embeddings(input_ids)
        ngram_pos_matrix = calculate_ngram_position_matrix(attn_mask=attention_mask,
                                                           max_ngram_size=self.max_n_gram_len,
                                                           model_device=self.device)
        embeds = torch.bmm(ngram_pos_matrix, embeds)
        embeds, cls_embeds  = self.bert(inputs_embeds=embeds)
        logit = self.out_layer(embeds[:,0,:])

        return logit

    def training_step(self, batch, batch_nb):
        # batch
        input_texts, batch_ids, label = batch

        # tokenize
        batch_inputs = self.tokenizer(list(input_texts), return_tensors="pt", padding=True)
        input_ids = batch_inputs['input_ids']
        attention_mask = batch_inputs['attention_mask']
        batch_labels = batch_labels.type('torch.FloatTensor')

        # fwd
        y_hat = self(input_ids, attention_mask)

        # loss
        loss = F.binary_cross_entropy_with_logits(y_hat, label)

        # logs
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # batch
        input_texts, batch_ids, label = batch

        # tokenize
        batch_inputs = self.tokenizer(list(input_texts), return_tensors="pt", padding=True)
        input_ids = batch_inputs['input_ids']
        attention_mask = batch_inputs['attention_mask']
        batch_labels = batch_labels.type('torch.FloatTensor')

        # fwd
        y_hat = self(input_ids, attention_mask)

        # loss
        loss = F.binary_cross_entropy_with_logits(y_hat, label)

        # acc
        a, y_hat = torch.max(y_hat, dim=1)
        val_acc = accuracy_score(y_hat.cpu(), label.cpu())
        val_acc = torch.tensor(val_acc)

        return {'val_loss': loss, 'val_acc': val_acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss, 'avg_val_acc': avg_val_acc}
        return {'val_loss': avg_loss, 'progress_bar': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, label = batch

        y_hat, attn = self(input_ids, attention_mask, token_type_ids)

        a, y_hat = torch.max(y_hat, dim=1)
        test_acc = accuracy_score(y_hat.cpu(), label.cpu())

        return {'test_acc': torch.tensor(test_acc)}

    def test_epoch_end(self, outputs):

        avg_test_acc = torch.stack([x['test_acc'] for x in outputs]).mean()

        tensorboard_logs = {'avg_test_acc': avg_test_acc}
        return {'avg_test_acc': avg_test_acc, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-05, eps=1e-08)

    def train_dataloader(self):
        return bert_mnli_train_dataloader

    def val_dataloader(self):
        return bert_mnli_val_dataloader

    def test_dataloader(self):
        return bert_mnli_test_dataloader
