import math

import torch
from torch import nn

from config.config import option as opt
from model.kan import KAN


def build_ffn(sizes):
    if opt.kan:
        return KAN(sizes)
    else:
        return nn.Sequential(
            nn.Linear(sizes[0], sizes[1]),
            nn.Dropout(0.1),
            nn.Linear(sizes[1], sizes[2])
        )


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)

        # self.linear1 = nn.Linear(d_model, d_model * 4)
        # self.dropout = nn.Dropout(0.1)
        # self.linear2 = nn.Linear(d_model * 4, d_model)

        sizes = [d_model, d_model * 4, d_model]
        self.ffn = build_ffn(sizes)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.ffn(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class CustomTransformerEncoder(nn.Module):
    def __init__(self, d_model, n_head, num_layers):
        super(CustomTransformerEncoder, self).__init__()
        encoder_layer = CustomTransformerEncoderLayer(d_model, n_head)
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return src


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=32):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # 调整位置编码的形状
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 假设 x 的形状为 (seq_len, batch_size, d_model)
        return x + self.pe[:x.size(0), :, :]
