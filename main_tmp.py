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

from utils import all_metrics, print_metrics, get_ngram_encoding

model_name = '../bert_base_uncased'
device = 'cuda:0'
num_epochs = 50
ngram_size = 32
batch_size_train = 32
batch_size_dev = 8
batch_size_test = 8
path = '/scratch/xl3119/Multi-Filter-Residual-Convolutional-Neural-Network/data/mimic3'
use_attention = False
multi_gpu = True
num_gpu = 3

tokenizer = AutoTokenizer.from_pretrained(model_name)

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
        batch_inputs = self.tokenizer([elem[0] for elem in batch], return_tensors="pt", padding=True)
        input_ids = batch_inputs["input_ids"]
        ngram_encoding = get_ngram_encoding(attn_mask=batch_inputs['attention_mask'],
                                            ngram_size=self.ngram_size,
                                            sep_cls=True)
        labels = torch.cat([elem[2] for elem in batch], dim=0).type('torch.FloatTensor')

        return (input_ids, ngram_encoding, labels)


class NGramTransformer(nn.Module):

    def __init__(self, model_name='', ngram_size = 32, n_class = 50):
        super().__init__()
        self.ngram_size = ngram_size
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        self.out_layer = nn.Linear(self.hidden_size, n_class)
        self.wd_emb = self.bert.embeddings.word_embeddings

    def forward(self, input_ids=None, ngram_encoding=None):
        embeds = torch.bmm(ngram_encoding, self.wd_emb(input_ids))
        embeds, cls_embeds  = self.bert(inputs_embeds=embeds)
        logits = self.out_layer(embeds[:,0,:])

        return logit

class NGramTransformer_Attn(nn.Module):

    def __init__(self, model_name='', ngram_size = 32, n_class = 50,device= 'cuda:0'):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        self.class_size = n_class
        self.ngram_size = ngram_size

        self.wd_emb = self.bert.embeddings.word_embeddings
        self.attn_layer = nn.Linear(self.hidden_size, self.class_size)
        self.out_layer = nn.Linear(self.hidden_size, 1)

    def forward(self, input_ids=None, ngram_encoding=None):
        embeds = torch.bmm(ngram_encoding, self.wd_emb(input_ids))
        embeds, cls_embeds  = self.bert(inputs_embeds=embeds)

        attn_weights = torch.transpose(self.attn_layer(embeds), 1, 2)
        attn_weights = F.softmax(attn_weights)
        attn_outputs = torch.bmm(attn_weights,embeds)
        logits = self.out_layer(attn_outputs)
        logits = logit.view(-1, self.class_size)

        return logit

def eval(model, tokenizer, dev_loader, device, ngram_size):
    model.eval()
    total_loss = 0.
    num_examples = 0
    criterion = torch.nn.BCEWithLogitsLoss()

    k = 5
    y = []
    yhat = []
    yhat_raw = []

    with torch.no_grad():
        for idx, (input_ids, ngram_encoding, labels) in enumerate(dev_loader):
            input_ids = input_ids.to(device)
            ngram_encoding = ngram_encoding.to(device)
            labels = labels.to(device)

            logits = model(input_ids, ngram_encoding)
            loss = criterion(logits, labels)
            total_loss += loss.item() * logits.size()[0]
            num_examples += logits.size()[0]

            y.append(labels.cpu().detach().numpy())
            yhat.append(np.round(torch.sigmoid(logits).cpu().detach().numpy()))
            yhat_raw.append(torch.sigmoid(logits).cpu().detach().numpy())

        y = np.concatenate(y, axis=0)
        yhat = np.concatenate(yhat, axis=0)
        yhat_raw = np.concatenate(yhat_raw, axis=0)
        metrics = all_metrics(yhat, y, k=k, yhat_raw=yhat_raw)
        metrics = all_metrics(yhat, y, k=k, yhat_raw=yhat_raw)
        print_metrics(metrics)

    print('Total eval loss after epoch is {}.'.format(str(total_loss / num_examples)))


def train(model_name, train_loader, device, ngram_size, num_epochs):
    if use_attention:
        model = NGramTransformer_Attn(model_name, ngram_size).to(device)
    else:
        model = NGramTransformer(model_name, ngram_size).to(device)

    if multi_gpu:
        device_ids = [i for i in range(num_gpu)]
        model = torch.nn.DataParallel(model, device_ids= device_ids)
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=2e-05, eps=1e-08)
    criterion = torch.nn.BCEWithLogitsLoss()
    model.zero_grad()

    for i in range(num_epochs):
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
        eval(model, tokenizer, dev_loader, device, ngram_size)
        torch.save(model, 'model.pt')

with open(join(path,'TOP_50_CODES.csv'),'r') as f:
    idx2code = [elem[:-1] for elem in f.readlines()]
code2idx = {elem:i for i, elem in enumerate(idx2code)}

train_df = pd.read_csv(join(path,'train_50.csv'))
dev_df = pd.read_csv(join(path,'dev_50.csv'))
test_df = pd.read_csv(join(path,'test_50.csv'))

train_texts = [elem[6:-6] for elem in train_df['TEXT']]
dev_texts = [elem[6:-6] for elem in dev_df['TEXT']]
test_texts = [elem[6:-6] for elem in test_df['TEXT']]

train_codes = [[code2idx[code] for code in elem.split(';')] for elem in train_df['LABELS']]
dev_codes = [[code2idx[code] for code in elem.split(';')] for elem in dev_df['LABELS']]
test_codes = [[code2idx[code] for code in elem.split(';')] for elem in test_df['LABELS']]

train_labels = [sum([torch.arange(50) == torch.Tensor([code]) for code in sample]) for sample in train_codes]
dev_labels = [sum([torch.arange(50) == torch.Tensor([code]) for code in sample]) for sample in dev_codes]
test_labels = [sum([torch.arange(50) == torch.Tensor([code]) for code in sample]) for sample in test_codes]

train_dataset = mimic3_dataset(train_texts, train_labels, tokenizer)
dev_dataset = mimic3_dataset(dev_texts, dev_labels, tokenizer)
test_dataset = mimic3_dataset(test_texts, test_labels, tokenizer)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size_train,
                          collate_fn=train_dataset.mimic3_col_func,
                          shuffle=True)
dev_loader = DataLoader(dataset=dev_dataset,
                        batch_size=batch_size_dev,
                        collate_fn=dev_dataset.mimic3_col_func,
                        shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size_test,
                         collate_fn=test_dataset.mimic3_col_func,
                         shuffle=True)

train(model_name, train_loader, device, ngram_size, num_epochs)
