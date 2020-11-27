import os
# for reproducibility, must before import torch
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # noqa

from datasets import load_dataset
import numpy as np
import random
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import torch
import argparse
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import (Trainer, TrainingArguments,
                          BertTokenizerFast, BertModel, BertPreTrainedModel)
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn
import datasets
from sklearn.datasets import fetch_20newsgroups
import util
import json
import logging
# logging.disable(logging.ERROR)


class RelatedEmbeddings(nn.Module):
    """Construct the embeddings from relatedness between words and labels."""

    def __init__(self, related_embeddings):
        super().__init__()
        self.relatedness = nn.Embedding.from_pretrained(related_embeddings,
                                                        freeze=False
                                                        )

    def forward(self, input_ids):
        relatedness = torch.mean(self.relatedness(input_ids), dim=1)
        return relatedness


class KilBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, aug=False, my_tokenizer=None, my_keywords=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.tokenizer = my_tokenizer
        self.keywords = tokenizer.tokenize(' '.join(my_keywords))
        self.aug = aug
        self.classifier = nn.Linear(config.hidden_size + len(self.keywords) * aug,
                                    config.num_labels)
        self.relatedness_embeddings = RelatedEmbeddings(self.get_relatedness())

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        if self.aug:
            # self.relatedness_embeddings = RelatedEmbeddings(self.get_relatedness()).cuda()
            relatedness = self.relatedness_embeddings(input_ids)
            comb = torch.cat((pooled_output, relatedness), dim=1)
            logits = self.classifier(comb)
        else:
            logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_relatedness(self):
        vocab = self.tokenizer.get_vocab()
        pretrained_emb = self.bert.embeddings.word_embeddings.weight
        keywords_matrix = [pretrained_emb[vocab[k]] for k in self.keywords]

        return self.create_relatedness_matrix(keywords_matrix, pretrained_emb)

    @staticmethod
    def create_relatedness_matrix(keywords_matrix, embeddings_matrix):
        r_embed = []
        for x in embeddings_matrix:
            r_embed.append([torch.dot(x, k) for k in keywords_matrix])
        return torch.tensor(r_embed)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }



if __name__ == "__main__":
    util.setup_seed(6)
    parser = argparse.ArgumentParser(description='Kil Bert Project')
    parser.add_argument('-d', '--data', help='data name', default='imdb',
                        choices=['agnews', 'imdb', 'newsgroup'])
    args = parser.parse_args()

    with open('settings.json', 'r', encoding='utf-8') as f:
        settings = json.load(f)
    config = settings["bert"][args.data]
    config["model_name"] = 'bert-base-uncased'
    tokenizer = BertTokenizerFast.from_pretrained(config["model_name"])

    train, test = util.get_data(args.data)
    train = train.map(lambda e: tokenizer(e['text'],
                                      truncation=True, padding='max_length', max_length=config["max_len"]), batched=True)
    train = train.map(lambda e: {'labels': e['label']}, batched=True)
    train.set_format(type='torch', columns=[
                    'input_ids', 'token_type_ids', 'attention_mask', 'labels'])

    test = test.map(lambda e: tokenizer(e['text'],
                                        truncation=True, padding='max_length', max_length=config["max_len"]), batched=True)
    test = test.map(lambda e: {'labels': e['label']}, batched=True)
    test.set_format(type='torch', columns=[
                    'input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    
    model = KilBertForSequenceClassification.from_pretrained(
        config["model_name"], num_labels=config["num_classes"], output_hidden_states=False, aug=False,
        my_tokenizer=tokenizer, my_keywords=config["keywords"])

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=config["epoch"],              # total number of training epochs
        per_device_train_batch_size=config["batch_size"],  # batch size per device during training
        per_device_eval_batch_size=config["batch_size"],   # batch size for evaluation
        evaluation_strategy="epoch",
        learning_rate=config["lr"],
        do_eval=True,
        logging_steps=100000
    )

    trainer = Trainer(
        # the instantiated 🤗 Transformers model to be trained
        model=model,
        args=training_args,                  # training arguments, defined above
        train_dataset=train,         # training dataset
        eval_dataset=test,             # evaluation dataset
        compute_metrics=compute_metrics,
    )
    trainer.train()

    del model
    del trainer
    model = KilBertForSequenceClassification.from_pretrained(
        config["model_name"], num_labels=config["num_classes"], output_hidden_states=False, aug=True,
        my_tokenizer=tokenizer, my_keywords=config["keywords"])
    print("Kil Bert mode ... ")
    trainer = Trainer(
        # the instantiated 🤗 Transformers model to be trained
        model=model,
        args=training_args,                  # training arguments, defined above
        train_dataset=train,         # training dataset
        eval_dataset=test,             # evaluation dataset
        compute_metrics=compute_metrics,
    )
    trainer.train()
    print(f'data: {args.data}, lr: {config["lr"]}, bs: {config["batch_size"]}, epoch: {config["epoch"]}')

