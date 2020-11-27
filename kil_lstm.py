import os
# for reproducibility, must before import torch
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # noqa
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import util
from statistics import mean
import json
import logging
# logging.disable(logging.ERROR)


class PreEmbeddings(nn.Module):
    """Construct the embeddings from pretrained embeddings."""

    def __init__(self, config, pretrained_embeddings):
        super().__init__()
        pretrained_embeddings = pretrained_embeddings.astype('float32')
        self.word_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(pretrained_embeddings))
        self.dropout = nn.Dropout(config["embed_dropout_prob"])

    def forward(self, input_ids, class_relatedness_ids=None):
        embeddings = self.word_embeddings(input_ids)
        embeddings = self.dropout(embeddings)
        return embeddings


class RelatedEmbeddings(nn.Module):
    """Construct the embeddings from relatedness between words and labels."""

    def __init__(self, config, related_embeddings):
        super().__init__()
        related_embeddings = related_embeddings.astype('float32')
        self.relatedness = nn.Embedding.from_pretrained(torch.from_numpy(related_embeddings))


    def forward(self, input_ids):
        relatedness = torch.mean(self.relatedness(input_ids), dim=1)
        return relatedness


class LSTMClassifier(torch.nn.Module):
    def __init__(self, config, pretrained_embeddings, related_embeddings):
        super().__init__()
        self.config = config
        self.word_embeddings = PreEmbeddings(config, pretrained_embeddings)
        self.relatedness = RelatedEmbeddings(config, related_embeddings)
        self.lstm = nn.LSTM(config["embed_dim"], config["embed_dim"]//2,
                            batch_first=True,
                            bidirectional=True,
                            num_layers=2
                            )
        self.fc1 = nn.Linear(
            config["embed_dim"]//2 + len(config['keywords']) * config['aug'], config["num_classes"])



    def forward(self, input_ids):
        word_embeddings = self.word_embeddings(input_ids)
        relatedness = self.relatedness(input_ids)
        lstm_out, (ht, ct) = self.lstm(word_embeddings)
        if self.config["aug"]:
            comb = torch.cat((ht[-1], relatedness), dim=1)
            x = self.fc1(comb)
        else:
            x = self.fc1(ht[-1])
        return x


def data_process(config):
    train_data, test_data = util.get_data(config['data_name'])

    vocab2index = util.get_vocab(
        train_data["text"] + test_data["text"], max_size=config["vocab_size"])

    train_data = train_data.map(lambda e: util.encode_sentence(
        e["text"], vocab2index, config))
    train_data.set_format(type='torch', columns=['input_ids', 'label'])
    test_data = test_data.map(lambda e: util.encode_sentence(
        e["text"], vocab2index, config))
    test_data.set_format(type='torch', columns=['input_ids', 'label'])
    train_dl = DataLoader(
        train_data, batch_size=config['batch_size'], shuffle=True)
    valid_dl = DataLoader(test_data, batch_size=config['batch_size'])

    pretrained_emb = util.load_glove('glove.6B.300d.txt')

    pretrained_embeddings = util.get_emb_matrix(
        pretrained_emb, vocab2index, emb_size=config['embed_dim'])
    keywords_matrix = [pretrained_emb[k] for k in config["keywords"]]
    related_embeddings = util.create_relatedness_matrix(
        keywords_matrix, pretrained_embeddings)

    print(f'embedding matrix shape: {pretrained_embeddings.shape}')
    print(f'relatedness matrix shape: {related_embeddings.shape}')

    return train_dl, valid_dl, pretrained_embeddings, related_embeddings


def get_res(config, train_dl, valid_dl, pretrained_embeddings, related_embeddings):
    model = LSTMClassifier(config, pretrained_embeddings, related_embeddings)
    model.cuda()
    top5, top1 = util.train_model(model, train_dl, valid_dl, config)
    del model
    return top5, top1


if __name__ == "__main__":
    util.setup_seed(6)
    parser = argparse.ArgumentParser(description='Knowledge in Labels Project')
    parser.add_argument('-d', '--data', help='data name', default='imdb',
                        choices=['agnews', 'imdb', 'newsgroup'])
    parser.add_argument('-g', '--gpu', help='gpu id', type=int, default=0)
    args = parser.parse_args()

    with open('settings.json', 'r', encoding='utf-8') as f:
        settings = json.load(f)
    config = settings["lstm"][args.data]
    config["epochs"] = 20
    config["embed_dropout_prob"] = 0.2
    config["vocab_size"] = None
    config["data_name"] = args.data
    config["embed_dim"] = 300
    torch.cuda.set_device(args.gpu)
    print(f'Using GPU #{torch.cuda.current_device()}: {torch.cuda.get_device_name()}')

    train_dl, valid_dl, pretrained_embeddings, related_embeddings = data_process(config)

    config['aug'] = False
    top5, top1 = get_res(
        config, train_dl, valid_dl, pretrained_embeddings, related_embeddings)

    print('using Kil Mode')
    config['aug'] = True
    top5_aug, top1_aug = get_res(
        config, train_dl, valid_dl, pretrained_embeddings, related_embeddings)

    print(f'data: {config["data_name"]}, lr: {config["lr"]}, \n top5: {top5:.6f}, top1: {top1:.6f},  \n top5_aug: {top5_aug:.6f}, top1_aug: {top1_aug:.6f}')

    
