import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import AutoTokenizer, AutoModel

from utils import all_metrics, print_metrics

model_name = 'bert-base-uncased'
device = 'cuda:0'
num_epochs = 10
max_n_gram_len = 64
batch_size_train = 32
batch_size_dev = 32
batch_size_test = 32
path = '/content/drive/My Drive'

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


def calculate_ngram_position_matrix(attn_mask = None, max_ngram_size = None, model_device = None):

    for i, elem in enumerate([attn_mask, max_ngram_size, model_device]):
        if elem is None:
            raise NameError('You must give input in position {}'.format(i))

    if attn_mask.device != model_device:
        attn_mask = attn_mask.to(model_device)

    batch_sent_lens = torch.sum(attn_mask,1)
    max_sent_len = torch.max(batch_sent_lens).item()
    num_n_grams = [math.ceil(elem / max_ngram_size) for elem in batch_sent_lens.tolist()]
    max_num_n_grams = max(num_n_grams)
    arange_t = torch.arange(max_sent_len)

    n_gram_pos = [[min(j * max_n_gram_len, batch_sent_lens[i].item()) for j in range(elem+1)] for i, elem in enumerate(num_n_grams)]
    for i in range(len(n_gram_pos)):
        n_gram_pos[i] = n_gram_pos[i] + [-1]*(max_num_n_grams+1-len(n_gram_pos[i]))
    n_gram_pos_matrix = [torch.cat([((arange_t>=elem[i])*(arange_t<elem[i+1])).unsqueeze(0) for i in range(max_num_n_grams)]).unsqueeze(0) for elem in n_gram_pos]
    n_gram_pos_matrix = torch.cat(n_gram_pos_matrix).to(model_device)

    return n_gram_pos_matrix.type(torch.FloatTensor).to(model_device)


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


def eval(model, tokenizer, dev_loader, device, max_n_gram_len):
    model.eval()
    total_loss = 0.
    num_examples = 0
    criterion = torch.nn.BCEWithLogitsLoss()

    k = 5
    y = []
    yhat = []
    yhat_raw = []

    with torch.no_grad():
        for idx, (batch_texts, batch_ids, batch_labels) in enumerate(dev_loader):

            batch_texts = list(batch_texts)
            batch_inputs = tokenizer(batch_texts, return_tensors="pt", padding=True)

            input_ids = batch_inputs['input_ids'].to(device)
            attention_mask = batch_inputs['attention_mask'].to(device)
            batch_labels = batch_labels.type('torch.FloatTensor').to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item() * outputs.size()[0]
            num_examples += outputs.size()[0]

            y.append(batch_labels.cpu().detach().numpy())
            yhat.append(np.round(torch.sigmoid(outputs).cpu().detach().numpy()))
            yhat_raw.append(torch.sigmoid(outputs).cpu().detach().numpy())

        y = np.concatenate(test_y, axis=0)
        yhat = np.concatenate(test_y_hat, axis=0)
        yhat_raw = np.concatenate(test_y_hat_raw, axis=0)
        metrics = all_metrics(yhat, y, k=k, yhat_raw=yhat_raw)
        metrics = all_metrics(yhat, y, k=k, yhat_raw=yhat_raw)
        print_metrics(metrics)

    print('Total eval loss after epoch is {}.'.format(str(total_loss / num_examples)))


def train(model_name, train_loader, device, max_n_gram_len, num_epochs):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = NGramTransformer(model_name, max_n_gram_len).to(device)
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=2e-05, eps=1e-08)
    criterion = torch.nn.BCEWithLogitsLoss()
    model.zero_grad()


    for i in range(num_epochs):
        total_loss = 0.
        num_examples = 0
        for idx, (batch_texts, batch_ids, batch_labels) in enumerate(train_loader):

            batch_texts = list(batch_texts)
            batch_inputs = tokenizer(batch_texts, return_tensors="pt", padding=True)

            input_ids = batch_inputs['input_ids'].to(device)
            attention_mask = batch_inputs['attention_mask'].to(device)
            batch_labels = batch_labels.type('torch.FloatTensor').to(device)

            model.train()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, batch_labels)

            loss.backward()
            optimizer.step()
            model.zero_grad()
            total_loss += loss.item() * outputs.size()[0]
            num_examples += outputs.size()[0]

        print('Average train loss after epoch {} is {}.'.format(str(i+1),str(total_loss / num_examples)))
        eval(model, tokenizer, dev_loader, device, max_n_gram_len)
        torch.save(model, 'model.pt')



with open(join(path,'TOP_50_CODES.csv'),'r') as f:
    idx2code = [elem[:-1] for elem in f.readlines()]
code2idx = {elem:i for i, elem in enumerate(idx2code)}

train_df = pd.read_csv(join(path,'train_50.csv'))
dev_df = pd.read_csv(join(path,'dev_50.csv'))
test_df = pd.read_csv(join(path,'test_50.csv'))

train_texts = [elem[6:-6] for elem in train_df['TEXT']]
train_codes = [[code2idx[code] for code in elem.split(';')] for elem in train_df['LABELS']]
train_labels = [sum([torch.arange(50) == torch.Tensor([code]) for code in sample]) for sample in train_codes]

dev_texts = [elem[6:-6] for elem in dev_df['TEXT']]
dev_codes = [[code2idx[code] for code in elem.split(';')] for elem in dev_df['LABELS']]
dev_labels = [sum([torch.arange(50) == torch.Tensor([code]) for code in sample]) for sample in dev_codes]

test_texts = [elem[6:-6] for elem in test_df['TEXT']]
test_codes = [[code2idx[code] for code in elem.split(';')] for elem in test_df['LABELS']]
test_labels = [sum([torch.arange(50) == torch.Tensor([code]) for code in sample]) for sample in test_codes]

train_dataset = mimic3_dataset(train_texts, train_labels)
dev_dataset = mimic3_dataset(dev_texts, dev_labels)
test_dataset = mimic3_dataset(test_texts, test_labels)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size_train,
                          shuffle=True)
dev_loader = DataLoader(dataset=dev_dataset,
                        batch_size=batch_size_dev,
                        shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size_test,
                         shuffle=True)

train(model_name, train_loader, device, max_n_gram_len, num_epochs)
eval(model, tokenizer, dev_loader, device, max_n_gram_len)
