"""DUET forecasting model adapted for the local TSLib/QAR interface.

Source architecture: https://github.com/decisionintelligence/DUET

The upstream repository is a standalone benchmark package.  This file keeps the
core DUET modules self-contained and returns only the forecast tensor so that it
works with ``exp_long_term_forecasting.py``.
"""

from __future__ import annotations

import math
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.distributions.normal import Normal
from torch.nn.functional import gumbel_softmax

from layers.Autoformer_EncDec import series_decomp


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


class LinearExtractor(nn.Module):
    def __init__(self, configs, individual: bool = False):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.d_model
        self.decomposition = series_decomp(configs.moving_avg)
        self.individual = individual
        self.channels = configs.enc_in
        self.enc_in = 1 if configs.CI else configs.enc_in
        if self.individual:
            self.linear_seasonal = nn.ModuleList()
            self.linear_trend = nn.ModuleList()
            for _ in range(self.channels):
                self.linear_seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.linear_trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.linear_trend = nn.Linear(self.seq_len, self.pred_len)
            self.linear_seasonal.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.linear_trend.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                dtype=seasonal_init.dtype,
                device=seasonal_init.device,
            )
            trend_output = torch.zeros_like(seasonal_output)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.linear_seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.linear_trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.linear_seasonal(seasonal_init)
            trend_output = self.linear_trend(trend_init)
        return (seasonal_output + trend_output).permute(0, 2, 1)

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        if x_enc.shape[0] == 0:
            return torch.empty((0, self.pred_len, self.enc_in), device=x_enc.device)
        return self.encoder(x_enc)[:, -self.pred_len :, :]


class DistributionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.distribution_fit = nn.Sequential(
            nn.Linear(config.seq_len, config.hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.num_experts, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.distribution_fit(torch.mean(x, dim=-1))


class SparseDispatcher:
    def __init__(self, num_experts: int, gates: torch.Tensor):
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp: torch.Tensor):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates: bool = True):
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            stitched = torch.einsum("i...,ij->i...", stitched, self._nonzero_gates)
        shape = list(expert_out[-1].shape)
        shape[0] = self._gates.size(0)
        zeros = torch.zeros(*shape, device=stitched.device)
        return zeros.index_add(0, self._batch_index, stitched.float())

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class LinearExtractorCluster(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.noisy_gating = config.noisy_gating
        self.num_experts = config.num_experts
        self.k = config.k
        self.experts = nn.ModuleList([LinearExtractor(config) for _ in range(self.num_experts)])
        self.W_h = nn.Parameter(torch.eye(self.num_experts))
        self.gate = DistributionEncoder(config)
        self.noise = DistributionEncoder(config)
        self.n_vars = config.enc_in
        self.revin = RevIN(self.n_vars)
        self.CI = config.CI
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert self.k <= self.num_experts

    @staticmethod
    def cv_squared(x: torch.Tensor) -> torch.Tensor:
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    @staticmethod
    def _gates_to_load(gates: torch.Tensor) -> torch.Tensor:
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        return torch.where(is_in, prob_if_in, prob_if_out)

    def noisy_top_k_gating(self, x: torch.Tensor, train: bool, noise_epsilon: float = 1e-2):
        clean_logits = self.gate(x)
        if self.noisy_gating and train:
            raw_noise_stddev = self.noise(x)
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            logits = noisy_logits @ self.W_h
        else:
            noisy_logits = clean_logits
            noise_stddev = None
            logits = clean_logits
        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)
        zeros = torch.zeros_like(logits)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        if self.noisy_gating and self.k < self.num_experts and train:
            load = self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x: torch.Tensor, loss_coef: float = 1.0):
        gates, load = self.noisy_top_k_gating(x, self.training)
        importance = gates.sum(0)
        loss = (self.cv_squared(importance) + self.cv_squared(load)) * loss_coef
        dispatcher = SparseDispatcher(self.num_experts, gates)
        if self.CI:
            x_norm = rearrange(x, "(b n) l c -> b l (n c)", n=self.n_vars)
            x_norm = self.revin(x_norm, "norm")
            x_norm = rearrange(x_norm, "b l (n c) -> (b n) l c", n=self.n_vars)
        else:
            x_norm = self.revin(x, "norm")
        expert_inputs = dispatcher.dispatch(x_norm)
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        return y, loss


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        _, _, _, e_dim = queries.shape
        scale = self.scale or 1.0 / sqrt(e_dim)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            large_negative = -math.log(1e10)
            attention_mask = torch.where(attn_mask == 0, large_negative, 0)
            scores = scores * attn_mask + attention_mask
        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        values = torch.einsum("bhls,bshd->blhd", attn, values)
        if self.output_attention:
            return values.contiguous(), attn
        return values.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        bsz, q_len, _ = queries.shape
        _, k_len, _ = keys.shape
        heads = self.n_heads
        queries = self.query_projection(queries).view(bsz, q_len, heads, -1)
        keys = self.key_projection(keys).view(bsz, k_len, heads, -1)
        values = self.value_projection(values).view(bsz, k_len, heads, -1)
        out, attn = self.inner_attention(queries, keys, values, attn_mask, tau=tau, delta=delta)
        out = out.view(bsz, q_len, -1)
        return self.out_projection(out), attn


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class MahalanobisMask(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        frequency_size = input_size // 2 + 1
        self.A = nn.Parameter(torch.randn(frequency_size, frequency_size), requires_grad=True)

    def calculate_prob_distance(self, x: torch.Tensor) -> torch.Tensor:
        xf = torch.abs(torch.fft.rfft(x, dim=-1))
        x1 = xf.unsqueeze(2)
        x2 = xf.unsqueeze(1)
        diff = x1 - x2
        temp = torch.einsum("dk,bxck->bxcd", self.A, diff)
        dist = torch.einsum("bxcd,bxcd->bxc", temp, temp)
        exp_dist = 1 / (dist + 1e-10)
        mask = (1 - torch.eye(exp_dist.shape[-1], device=exp_dist.device)).repeat(exp_dist.shape[0], 1, 1)
        exp_dist = torch.einsum("bxc,bxc->bxc", exp_dist, mask)
        exp_max, _ = torch.max(exp_dist, dim=-1, keepdim=True)
        exp_max = exp_max.detach()
        p = exp_dist / exp_max
        identity = torch.eye(p.shape[-1], device=p.device).repeat(p.shape[0], 1, 1)
        p1 = torch.einsum("bxc,bxc->bxc", p, mask)
        return (p1 + identity) * 0.99

    @staticmethod
    def bernoulli_gumbel_rsample(distribution_matrix: torch.Tensor) -> torch.Tensor:
        bsz, channels, dims = distribution_matrix.shape
        flat = rearrange(distribution_matrix, "b c d -> (b c d) 1")
        r_flat = 1 - flat
        logits = torch.concat([torch.log(flat / r_flat), torch.log(r_flat / flat)], dim=-1)
        resample = gumbel_softmax(logits, hard=True)
        return rearrange(resample[..., 0], "(b c d) -> b c d", b=bsz, c=channels, d=dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.calculate_prob_distance(x)
        sample = self.bernoulli_gumbel_rsample(p)
        return sample.unsqueeze(1)


class DUETCore(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cluster = LinearExtractorCluster(config)
        self.CI = config.CI
        self.n_vars = config.enc_in
        self.mask_generator = MahalanobisMask(config.seq_len)
        self.channel_transformer = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            True,
                            config.factor,
                            attention_dropout=config.dropout,
                            output_attention=config.output_attention,
                        ),
                        config.d_model,
                        config.n_heads,
                    ),
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for _ in range(config.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(config.d_model),
        )
        self.linear_head = nn.Sequential(nn.Linear(config.d_model, config.pred_len), nn.Dropout(config.fc_dropout))

    def forward(self, x: torch.Tensor):
        if self.CI:
            channel_independent_input = rearrange(x, "b l n -> (b n) l 1")
            reshaped_output, l_importance = self.cluster(channel_independent_input)
            temporal_feature = rearrange(reshaped_output, "(b n) l 1 -> b l n", b=x.shape[0])
        else:
            temporal_feature, l_importance = self.cluster(x)
        temporal_feature = rearrange(temporal_feature, "b d n -> b n d")
        if self.n_vars > 1:
            changed_input = rearrange(x, "b l n -> b n l")
            channel_mask = self.mask_generator(changed_input)
            channel_group_feature, _ = self.channel_transformer(x=temporal_feature, attn_mask=channel_mask)
            output = self.linear_head(channel_group_feature)
        else:
            output = self.linear_head(temporal_feature)
        output = rearrange(output, "b n d -> b d n")
        output = self.cluster.revin(output, "denorm")
        return output, l_importance


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # DUET-specific defaults not present in the shared run.py parser.
        configs.CI = bool(int(getattr(configs, "CI", 1)))
        configs.num_experts = int(getattr(configs, "num_experts", 4))
        configs.k = int(getattr(configs, "k", 1))
        configs.noisy_gating = bool(int(getattr(configs, "noisy_gating", 1)))
        configs.hidden_size = int(getattr(configs, "hidden_size", configs.d_model))
        configs.fc_dropout = float(getattr(configs, "fc_dropout", configs.dropout))
        configs.output_attention = bool(int(getattr(configs, "output_attention", 0)))
        self.model = DUETCore(configs)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        out, _ = self.model(x_enc)
        return out
