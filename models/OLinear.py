"""QAR-ready OLinear adapter.

The upstream OLinear code requires pre-generated orthogonal Q matrices plus
custom encoder layers.  For this benchmark matrix we provide a runnable adapter
that follows the same high-level recipe (RevIN -> token embedding -> temporal
orthogonal/linear block -> linear head) while defaulting the Q transform to the
identity/no-Q path.  This makes the row reproducible on the QAR compact caches
without adding dataset-specific Q-matrix generation.

Source family: https://github.com/jackyue1994/OLinear
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self.mean = x.mean(dim=tuple(range(1, x.ndim - 1)), keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=tuple(range(1, x.ndim - 1)), keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        if mode == "denorm":
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
            return x * self.stdev + self.mean
        raise ValueError(f"Unsupported RevIN mode: {mode}")


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(configs.enc_in)
        self.seq_len = int(configs.seq_len)
        self.d_model = int(configs.d_model)
        self.d_ff = int(configs.d_ff)
        self.embed_size = int(getattr(configs, "embed_size", 1))
        self.embed_size = max(1, self.embed_size)

        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.revin_layer = RevIN(self.enc_in, affine=True)
        self.dropout = nn.Dropout(float(getattr(configs, "dropout", 0.1)))

        self.input_proj = nn.Linear(self.seq_len * self.embed_size, self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(getattr(configs, "n_heads", 8)),
            dim_feedforward=self.d_ff,
            dropout=float(getattr(configs, "dropout", 0.1)),
            activation=str(getattr(configs, "activation", "gelu")),
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(getattr(configs, "e_layers", 2)))
        self.output_proj = nn.Linear(self.d_model, self.pred_len * self.embed_size)
        self.fc = nn.Sequential(
            nn.Linear(self.pred_len * self.embed_size, self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff, self.pred_len),
        )

    def token_emb(self, x: torch.Tensor) -> torch.Tensor:
        if self.embed_size <= 1:
            return x.transpose(-1, -2).unsqueeze(-1)
        return x.transpose(-1, -2).unsqueeze(-1) * self.embeddings

    def temporal_block(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, T, D]
        bsz, channels, _time, emb = x.shape
        x = x.flatten(-2)
        x = self.input_proj(x)
        x = self.encoder(x)
        x = self.output_proj(x)
        return x.reshape(bsz, channels, self.pred_len, emb)

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        x = self.revin_layer(x, mode="norm")
        x = self.token_emb(x)
        x = self.temporal_block(x)
        out = self.fc(x.flatten(-2)).transpose(-1, -2)
        out = self.dropout(out)
        out = self.revin_layer(out, mode="denorm")
        return out
