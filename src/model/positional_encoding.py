import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, model_dimension, dropout, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.position = torch.zeros((1, max_len, model_dimension))
        temp_encoding = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1)
        temp_encoding /= torch.pow(10000, torch.arange(0, model_dimension, 2, dtype=torch.float32) / model_dimension)
        self.position[:, :, 0::2] = torch.sin(temp_encoding)
        self.position[:, :, 1::2] = torch.cos(temp_encoding)

    def forward(self, x):
        x = x + self.position[:, :x.shape(1), :].to(x.device)
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, model_dimension, dropout, max_len=500):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.position_embedding = nn.Embedding(max_len, model_dimension)

    def forward(self, x, position):
        x = x + self.position_embedding(position)
        return self.dropout(x)
