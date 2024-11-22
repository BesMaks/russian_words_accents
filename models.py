import json
import os
from typing import Union

import torch
from torch import nn


class BertNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = json.load(open(os.path.join(os.getcwd(), "params.json")))[
            "BertNet"
        ]
        embedding_dim = self.get_parameter("EMBEDDING_DIM")
        dict_size = self.get_parameter("DICT_SIZE")
        max_length = self.get_parameter("MAX_LENGTH")
        self.embedding_layer = nn.Embedding(
            num_embeddings=dict_size, embedding_dim=embedding_dim
        )
        self.dropout = nn.Dropout(0.2)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embedding_dim, 1, batch_first=True, dropout=0.2),
            2,
        )
        self.extra_dense = nn.Sequential(
            nn.Linear(embedding_dim * max_length, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, max_length),
        )

    def get_parameter(self, key: str) -> Union[int, float]:
        return self.params[key]

    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.dropout(x)
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1, end_dim=2)
        x = self.extra_dense(x)
        return x


class DenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = json.load(open(os.path.join(os.getcwd(), "params.json")))[
            "DenseNet"
        ]
        embedding_dim = self.get_parameter("EMBEDDING_DIM")
        dict_size = self.get_parameter("DICT_SIZE")
        max_length = self.get_parameter("MAX_LENGTH")
        self.embedding_layer = nn.Embedding(
            num_embeddings=dict_size, embedding_dim=embedding_dim
        )
        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Sequential(
            nn.Linear(embedding_dim * max_length, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, max_length),
        )

    def get_parameter(self, key: str) -> Union[int, float]:
        return self.params[key]

    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1, end_dim=2)
        x = self.dense(x)
        return x

# 
class LemmaUsingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = json.load(open(os.path.join(os.getcwd(), "params.json")))[
            "LemmaUsingNet"
        ]
        embedding_dim = self.get_parameter("EMBEDDING_DIM")
        dict_size = self.get_parameter("DICT_SIZE")
        lemma_dict_size = self.get_parameter("LEMMA_DICT_SIZE")
        max_length = self.get_parameter("MAX_LENGTH")
        max_lemma_length = self.get_parameter("MAX_LEMMA_LENGTH")
        self.word_embeddings = nn.Embedding(dict_size, embedding_dim)
        self.lemma_embedding = nn.Embedding(lemma_dict_size, embedding_dim)
        self.dropout = nn.Dropout(0.2)
        self.dense = nn.Sequential(
            nn.Linear(embedding_dim * (max_length + max_lemma_length), 1024),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(512, max_length),
        )

    def get_parameter(self, key: str) -> Union[int, float]:
        return self.params[key]

    def forward(self, x, lemma):
        x = self.word_embeddings(x)
        x = self.dropout(x)
        lemma = self.lemma_embedding(lemma)
        lemma = self.dropout(lemma)
        x = torch.cat((x, lemma), 1)
        x = torch.flatten(x, start_dim=1, end_dim=2)
        x = self.dense(x)
        return x


class LemmaUsingNetWithPositionalEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = json.load(open(os.path.join(os.getcwd(), "params.json")))[
            "LemmaUsingNet"
        ]
        embedding_dim = self.get_parameter("EMBEDDING_DIM")
        dict_size = self.get_parameter("DICT_SIZE")
        lemma_dict_size = self.get_parameter("LEMMA_DICT_SIZE")
        max_length = self.get_parameter("MAX_LENGTH")
        max_lemma_length = self.get_parameter("MAX_LEMMA_LENGTH")
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embedding_dim, 1, batch_first=True, dropout=0.2),
            2,
        )
        self.word_embeddings = nn.Embedding(dict_size, embedding_dim)
        self.word_pos_embeddings = nn.Embedding(max_length, embedding_dim)
        self.lemma_embedding = nn.Embedding(lemma_dict_size, embedding_dim)
        self.lemma_pos_embedding = nn.Embedding(max_lemma_length, embedding_dim)
        self.dropout = nn.Dropout(0.2)
        self.dense = nn.Sequential(
            nn.Linear(embedding_dim * (max_length + max_lemma_length), 1024),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(512, max_length),
        )

    def get_parameter(self, key: str) -> Union[int, float]:
        return self.params[key]

    def forward(self, x, lemma):
        x = self.word_embeddings(x)
        pos_embedding = self.word_pos_embeddings(
            torch.tensor([[i for i in range(len(x[0]))]] * len(x))
        )
        x = self.dropout(x)
        lemma = self.lemma_embedding(lemma)
        pos_lemma_embedding = self.lemma_pos_embedding(
            torch.tensor([[i for i in range(len(lemma[0]))]] * len(lemma))
        )
        lemma = self.dropout(lemma)
        x = torch.cat((x + pos_embedding, lemma + pos_lemma_embedding), 1)
        x = torch.flatten(x, start_dim=1, end_dim=2)
        x = self.dense(x)
        return x
