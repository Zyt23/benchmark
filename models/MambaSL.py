"""MambaSL fallback adapter for QAR classification.

The original MambaSL model depends on ``mamba_ssm``.  That CUDA extension is
not available in the current server environment, so this adapter provides a
lightweight Mamba-style gated temporal mixing block that can run with plain
PyTorch and the existing QAR classification loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GatedTemporalBlock(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dropout: float):
        super().__init__()
        padding = kernel_size // 2
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding, groups=d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        u = self.depthwise(u.transpose(1, 2)).transpose(1, 2)
        u = self.act(u) * torch.sigmoid(gate)
        return residual + self.dropout(self.out_proj(u))


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.num_class = int(getattr(configs, "num_class", 2))
        self.d_model = int(getattr(configs, "d_model", 64))
        dropout = float(getattr(configs, "dropout", 0.1))
        self.input_proj = nn.Linear(int(configs.enc_in), self.d_model)
        self.blocks = nn.Sequential(
            *[
                GatedTemporalBlock(
                    self.d_model,
                    kernel_size=max(3, int(getattr(configs, "d_conv", 4)) * 2 + 1),
                    dropout=dropout,
                )
                for _ in range(max(1, int(getattr(configs, "e_layers", 2))))
            ]
        )
        self.norm = nn.LayerNorm(self.d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.d_model, self.num_class))

    def classification(self, x_enc: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x_enc)
        x = self.blocks(x)
        x = self.norm(x)
        # mean + last pooling keeps both global and terminal state information.
        pooled = 0.5 * (x.mean(dim=1) + x[:, -1, :])
        return self.head(pooled)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == "classification":
            return self.classification(x_enc)
        raise NotImplementedError("MambaSL fallback adapter currently supports classification only")
