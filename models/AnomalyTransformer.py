"""Anomaly Transformer adapted to the Time-Series-Library model interface.

The architecture follows the official THUML implementation:
https://github.com/thuml/Anomaly-Transformer

Unlike the historical plain-Transformer reconstruction wrapper, this module
returns the learned series association and Gaussian prior association required
by the paper's minimax objective and association-discrepancy anomaly score.
"""

from __future__ import annotations

import math
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnomalyAttention(nn.Module):
    def __init__(self, win_size, scale=None, attention_dropout=0.0, output_attention=True):
        super().__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        positions = torch.arange(win_size, dtype=torch.float32)
        self.register_buffer("distances", torch.abs(positions[:, None] - positions[None, :]))

    def forward(self, queries, keys, values, sigma, attn_mask=None):
        _, length, _, head_dim = queries.shape
        scale = self.scale or 1.0 / sqrt(head_dim)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        series = self.dropout(torch.softmax(scale * scores, dim=-1))

        sigma = sigma.transpose(1, 2)
        sigma = torch.pow(3.0, torch.sigmoid(sigma * 5.0) + 1e-5) - 1.0
        sigma = sigma.unsqueeze(-1).expand(-1, -1, -1, length)
        distance = self.distances[:length, :length].unsqueeze(0).unsqueeze(0)
        prior = (1.0 / (math.sqrt(2.0 * math.pi) * sigma)) * torch.exp(
            -(distance ** 2) / (2.0 * sigma ** 2)
        )
        output = torch.einsum("bhls,bshd->blhd", series, values)
        if self.output_attention:
            return output.contiguous(), series, prior, sigma
        return output.contiguous(), None, None, None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super().__init__()
        d_keys = d_keys or d_model // n_heads
        d_values = d_values or d_model // n_heads
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model, n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None):
        batch, q_len, _ = queries.shape
        _, k_len, _ = keys.shape
        heads = self.n_heads
        q = self.query_projection(queries).view(batch, q_len, heads, -1)
        k = self.key_projection(keys).view(batch, k_len, heads, -1)
        v = self.value_projection(values).view(batch, k_len, heads, -1)
        sigma = self.sigma_projection(queries).view(batch, q_len, heads)
        output, series, prior, sigma = self.inner_attention(q, k, v, sigma, attn_mask)
        output = output.view(batch, q_len, -1)
        return self.out_projection(output), series, prior, sigma


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-(math.log(10000.0) / d_model))
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        self.token_conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            bias=False,
        )
        nn.init.kaiming_normal_(self.token_conv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        return self.token_conv(x.permute(0, 2, 1)).transpose(1, 2)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.0):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.value_embedding(x) + self.position_embedding(x))


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="gelu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, series, prior, sigma = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(1, 2))))
        y = self.dropout(self.conv2(y).transpose(1, 2))
        return self.norm2(x + y), series, prior, sigma


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        series_list, prior_list, sigma_list = [], [], []
        for layer in self.layers:
            x, series, prior, sigma = layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)
        if self.norm is not None:
            x = self.norm(x)
        return x, series_list, prior_list, sigma_list


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        win_size = int(configs.seq_len)
        enc_in = int(configs.enc_in)
        c_out = int(getattr(configs, "c_out", enc_in))
        d_model = int(getattr(configs, "d_model", 64))
        n_heads = int(getattr(configs, "n_heads", 4))
        while n_heads > 1 and d_model % n_heads:
            n_heads -= 1
        e_layers = int(getattr(configs, "e_layers", 2))
        d_ff = int(getattr(configs, "d_ff", 128))
        dropout = float(getattr(configs, "dropout", 0.0))

        self.embedding = DataEmbedding(enc_in, d_model, dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(
                            win_size,
                            attention_dropout=dropout,
                            output_attention=True,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation="gelu",
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def anomaly_detection(self, x_enc):
        encoded = self.embedding(x_enc)
        encoded, series, prior, sigma = self.encoder(encoded)
        reconstruction = self.projection(encoded)
        return reconstruction, series, prior, sigma

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name != "anomaly_detection":
            raise NotImplementedError("AnomalyTransformer supports anomaly_detection only")
        return self.anomaly_detection(x_enc)
