"""PyTorch LITE-style classifier for QAR compact classification.

LITE in aeon is TensorFlow/Keras based.  The server environment does not have
TensorFlow, so this module provides a QAR-ready PyTorch implementation inspired
by the LITE idea: lightweight multi-scale temporal convolutions with residual
connections and global average pooling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LiteBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float):
        super().__init__()
        k1 = max(3, int(kernel_size) | 1)
        k2 = max(3, (int(kernel_size) // 2) | 1)
        k3 = max(3, (int(kernel_size) // 4) | 1)
        branch_channels = max(1, out_channels // 4)
        self.bottleneck = nn.Conv1d(in_channels, branch_channels, kernel_size=1)
        self.conv1 = nn.Conv1d(branch_channels, branch_channels, kernel_size=k1, padding=k1 // 2, groups=branch_channels)
        self.conv2 = nn.Conv1d(branch_channels, branch_channels, kernel_size=k2, padding=k2 // 2, groups=branch_channels)
        self.conv3 = nn.Conv1d(branch_channels, branch_channels, kernel_size=k3, padding=k3 // 2, groups=branch_channels)
        self.pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
        )
        merged = branch_channels * 4
        self.mix = nn.Conv1d(merged, out_channels, kernel_size=1)
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bottleneck(x)
        y = torch.cat([self.conv1(z), self.conv2(z), self.conv3(z), self.pool(x)], dim=1)
        y = self.mix(y)
        y = self.bn(y + self.residual(x))
        return self.dropout(F.relu(y))


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.num_class = int(getattr(configs, "num_class", 2))
        in_channels = int(configs.enc_in)
        width = int(getattr(configs, "d_model", 64))
        width = max(16, width)
        kernel_size = int(getattr(configs, "lite_kernel_size", 40))
        dropout = float(getattr(configs, "dropout", 0.1))
        layers = max(2, int(getattr(configs, "e_layers", 2)))

        blocks = []
        channels = in_channels
        for i in range(layers):
            blocks.append(LiteBlock(channels, width, max(7, kernel_size // (2 ** i)), dropout))
            channels = width
        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Linear(width, self.num_class)

    def classification(self, x_enc: torch.Tensor) -> torch.Tensor:
        # x_enc: [B, T, C] -> [B, C, T]
        x = x_enc.transpose(1, 2)
        x = self.backbone(x)
        x = x.mean(dim=-1)
        return self.head(x)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == "classification":
            return self.classification(x_enc)
        raise NotImplementedError("LITE adapter currently supports classification only")
