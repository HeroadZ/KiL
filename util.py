import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import spacy
import re
import typing
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import random
from collections import Counter
from datasets import load_dataset
from statistics import mean
import datasets
from sklearn.datasets import fetch_20newsgroups

tok = spacy.load('en_core_web_sm')


def tokenize(text: str) -> [str]:
    text = re.sub(r"[^\x00-\x7F]+", " ", text.lower())
    return [token.text for token in tok.tokenizer(text) if token.text.isalnum()]


def get_data(name: str) -> tuple:
    if name == 'agnews':
        return load_dataset('ag_news', split=['train', 'test'])
    elif name == 'imdb':
        return load_dataset('imdb', split=['train', 'test'])
    elif name == 'newsgroup':
        train = fetch_20newsgroups(subset='train', shuffle=True)
        test = fetch_20newsgroups(subset='test')
        train_data = datasets.Dataset.from_dict({
            'text': train.data,
            'label': train.target
        })
        test_data = datasets.Dataset.from_dict({
            'text': test.data,
            'label': test.target
        })
        return train_data, test_data
    else:
        raise ValueError('no corresponding dataset')


def get_vocab(texts: [str], max_size: int=None) -> typing.Dict[str, int]:
    vocab = Counter()
    length = []
    for row in texts:
        tokens = tokenize(row)
        length.append(len(tokens))
        vocab.update(tokens)
    average_len = sum(length) / len(length)
    print(f'average length: {average_len}, max_len: {max(length)}')
    vocab2index = {"": 0, "UNK": 1}
    for word, _ in vocab.most_common(max_size):
        vocab2index[word] = len(vocab2index)
    return vocab2index


def encode_sentence(text: str, vocab2index: typing.Dict[str, int],
                    # pretrained_embed: typing.Dict[str, np.ndarray], labels: [np.ndarray], 
                    config: dict) -> dict:
    tokenized = tokenize(text)
    input_ids = [0] * config["max_len"]
    enc1 = [vocab2index.get(word, vocab2index["UNK"]) for word in tokenized]
    length = min(config["max_len"], len(enc1))
    input_ids[:length] = enc1[:length]

    return {
        'input_ids': input_ids, 
    }


def setup_seed(seed: int) -> None:
    CUDA = torch.cuda.is_available()
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if CUDA:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True



def load_glove(File: str) -> typing.Dict[str, np.ndarray]:
    print("Loading Glove Model")
    f = open(File, 'r', encoding='utf-8', errors='ignore')
    Embed = {}
    for line in f:
        splitLines = line.split()
        Embed[splitLines[0]] = np.array(
            [float(value) for value in splitLines[1:]])
    print(f'shape of embedding: ({len(Embed)}, {len(Embed["the"])})')
    return Embed


def get_emb_matrix(pretrained: typing.Dict[str, np.ndarray], vocab2index: typing.Dict[str, int],
                   emb_size: int = 300) -> np.ndarray:
    """ Creates embedding matrix from word vectors"""
    W = np.zeros((len(vocab2index), emb_size), dtype="float64")
    W[0] = np.zeros(emb_size, dtype='float64')  # adding a vector for padding
    # adding a vector for unknown words
    W[1] = np.random.uniform(-0.25, 0.25, emb_size)
    for word, i in vocab2index.items():
        if i in [0, 1]:
            continue
        if word in pretrained:
            W[i] = pretrained[word]
        else:
            W[i] = np.random.uniform(-0.25, 0.25, emb_size)
    return W


def train_model(model, train_dl, valid_dl, config: dict):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config['lr'])
    accs = []
    for i in range(config['epochs']):
        model.train()
        sum_loss = 0.0
        total = 0
        for batch in train_dl:
            input_ids = batch['input_ids'].long().cuda()
            y = batch['label'].long().cuda()
            y_pred = model(input_ids)
            optimizer.zero_grad()
            loss = F.cross_entropy(y_pred, y)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()*y.shape[0]
            total += y.shape[0]
        model.eval()
        val_sum_loss = 0.0
        val_total = 0
        preds, truth = [], []
        for batch in valid_dl:
            input_ids = batch['input_ids'].long().cuda()
            y = batch['label'].long().cuda()
            y_hat = model(input_ids)
            loss = F.cross_entropy(y_hat, y)
            preds += torch.max(y_hat, 1)[1].detach().cpu()
            truth += y.detach().cpu()
            val_total += y.shape[0]
            val_sum_loss += loss.item()*y.shape[0]
        val_loss, val_acc = val_sum_loss / val_total, accuracy_score(truth, preds)
        accs.append(val_acc)
        # print("epoch %d: train loss %.6f, val loss %.6f, val accuracy %.6f" %
        #       (i+1, sum_loss/total, val_loss, val_acc))

    return mean(sorted(accs)[-5:]), max(accs)


def cal_relatedness(v1: np.ndarray, v2: np.ndarray) -> float:
    return np.dot(v1, v2)


def create_relatedness_matrix(keywords_matrix: [np.ndarray],
                              embeddings_matrix: np.ndarray) -> np.ndarray:
    r_embed = []
    for x in embeddings_matrix:
        r_embed.append([cal_relatedness(x, k) for k in keywords_matrix])
    return np.array(r_embed)
