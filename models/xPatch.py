"""xPatch forecasting model adapted for the local TSLib/QAR interface.

Source architecture: https://github.com/stitsyuk/xPatch

This file is self-contained on purpose: the upstream implementation expects
``layers.decomp/network/revin`` modules, while this benchmark repository keeps
model additions under ``models/`` to avoid broad namespace changes.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True, subtract_last: bool = False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._get_statistics(x)
            x = x - (self.last if self.subtract_last else self.mean)
            x = x / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        if mode == "denorm":
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
            x = x * self.stdev
            x = x + (self.last if self.subtract_last else self.mean)
            return x
        raise ValueError(f"Unsupported RevIN mode: {mode}")

    def _get_statistics(self, x: torch.Tensor) -> None:
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1:, :].detach()
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()


class DECOMP(nn.Module):
    def __init__(self, ma_type: str = "reg", alpha: float = 0.1, beta: float = 0.1):
        super().__init__()
        self.ma_type = ma_type
        self.alpha = float(alpha)
        self.beta = float(beta)

    @staticmethod
    def _ema(x: torch.Tensor, alpha: float) -> torch.Tensor:
        out = torch.zeros_like(x)
        out[:, 0, :] = x[:, 0, :]
        for i in range(1, x.shape[1]):
            out[:, i, :] = alpha * x[:, i, :] + (1.0 - alpha) * out[:, i - 1, :]
        return out

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.ma_type == "ema":
            trend = self._ema(x, self.alpha)
        elif self.ma_type == "dema":
            ema1 = self._ema(x, self.alpha)
            ema2 = self._ema(ema1, self.beta)
            trend = 2 * ema1 - ema2
        else:
            trend = torch.zeros_like(x)
        seasonal = x - trend
        return seasonal, trend


class Network(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, patch_len: int, stride: int, padding_patch: str):
        super().__init__()
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.dim = patch_len * patch_len
        self.patch_num = (seq_len - patch_len) // stride + 1
        if padding_patch == "end":
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1

        self.fc1 = nn.Linear(patch_len, self.dim)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)

        self.conv1 = nn.Conv1d(self.patch_num, self.patch_num, patch_len, patch_len, groups=self.patch_num)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(self.patch_num)

        self.fc2 = nn.Linear(self.dim, patch_len)

        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(self.patch_num)

        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(self.patch_num * patch_len, pred_len * 2)
        self.gelu4 = nn.GELU()
        self.fc4 = nn.Linear(pred_len * 2, pred_len)

        self.fc5 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)

        self.fc6 = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len // 2)

        self.fc7 = nn.Linear(pred_len // 2, pred_len)
        self.fc8 = nn.Linear(pred_len * 2, pred_len)

    def forward(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        s = s.permute(0, 2, 1)
        t = t.permute(0, 2, 1)

        bsz, channels, input_len = s.shape
        s = torch.reshape(s, (bsz * channels, input_len))
        t = torch.reshape(t, (bsz * channels, input_len))

        if self.padding_patch == "end":
            s = self.padding_patch_layer(s)
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        s = self.fc1(s)
        s = self.gelu1(s)
        s = self.bn1(s)

        res = s
        s = self.conv1(s)
        s = self.gelu2(s)
        s = self.bn2(s)
        s = s + self.fc2(res)

        s = self.conv2(s)
        s = self.gelu3(s)
        s = self.bn3(s)

        s = self.flatten1(s)
        s = self.fc3(s)
        s = self.gelu4(s)
        s = self.fc4(s)

        t = self.fc5(t)
        t = self.avgpool1(t)
        t = self.ln1(t)
        t = self.fc6(t)
        t = self.avgpool2(t)
        t = self.ln2(t)
        t = self.fc7(t)

        x = torch.cat((s, t), dim=1)
        x = self.fc8(x)
        x = torch.reshape(x, (bsz, channels, self.pred_len))
        return x.permute(0, 2, 1)


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        seq_len = configs.seq_len
        pred_len = configs.pred_len
        c_in = configs.enc_in

        patch_len = int(getattr(configs, "patch_len", 16))
        patch_len = max(2, min(patch_len, seq_len))
        stride = int(getattr(configs, "stride", max(1, patch_len // 2)))
        padding_patch = getattr(configs, "padding_patch", "end")

        self.revin = bool(int(getattr(configs, "revin", 1)))
        self.revin_layer = RevIN(c_in, affine=True, subtract_last=False)

        self.ma_type = getattr(configs, "ma_type", "reg")
        alpha = float(getattr(configs, "alpha", 0.1))
        beta = float(getattr(configs, "beta", 0.1))
        self.decomp = DECOMP(self.ma_type, alpha, beta)
        self.net = Network(seq_len, pred_len, patch_len, stride, padding_patch)

    def forecast(self, x: torch.Tensor) -> torch.Tensor:
        if self.revin:
            x = self.revin_layer(x, "norm")
        if self.ma_type == "reg":
            x = self.net(x, x)
        else:
            seasonal_init, trend_init = self.decomp(x)
            x = self.net(seasonal_init, trend_init)
        if self.revin:
            x = self.revin_layer(x, "denorm")
        return x

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.forecast(x_enc)
