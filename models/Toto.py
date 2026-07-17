import os

import torch
import torch.nn.functional as F
from torch import nn


def _patch_torch_sdpa_for_toto() -> None:
    """Make Toto's enable_gqa call work on older PyTorch builds.

    Toto 2.0 calls ``scaled_dot_product_attention(..., enable_gqa=...)``.
    Some deployed PyTorch 2.4 builds do not expose that keyword yet.  When GQA
    is requested, emulate it by repeating K/V heads to match Q heads.
    """
    if getattr(F.scaled_dot_product_attention, "_toto_gqa_compat", False):
        return
    original_sdpa = F.scaled_dot_product_attention

    def compat_sdpa(query, key, value, *args, enable_gqa=False, **kwargs):
        if enable_gqa and query.shape[-3] != key.shape[-3]:
            if query.shape[-3] % key.shape[-3] != 0:
                raise ValueError(
                    "Cannot emulate GQA: query heads {} not divisible by key heads {}".format(
                        query.shape[-3], key.shape[-3]
                    )
                )
            repeat = query.shape[-3] // key.shape[-3]
            key = key.repeat_interleave(repeat, dim=-3)
            value = value.repeat_interleave(repeat, dim=-3)
        return original_sdpa(query, key, value, *args, **kwargs)

    compat_sdpa._toto_gqa_compat = True
    F.scaled_dot_product_attention = compat_sdpa


class Model(nn.Module):
    """Datadog Toto 2.0 zero-shot forecasting adapter.

    Requires:
        pip install toto-models

    Default checkpoint:
        Datadog/Toto-2.0-22m

    Input expected by Toto2Model is (batch, n_variates, time_steps).  The QAR
    loader provides (batch, time_steps, n_variates), so we transpose before and
    after forecasting.  A light per-series standardization is applied to match
    the other local foundation-model adapters.
    """

    def __init__(self, configs):
        super().__init__()
        try:
            from toto2 import Toto2Model
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise ImportError(
                "Toto 2.0 requires the optional package `toto-models` "
                "(which installs `toto2`). Install with: pip install toto-models"
            ) from exc

        _patch_torch_sdpa_for_toto()
        model_path = os.environ.get("TOTO2_MODEL_PATH", "Datadog/Toto-2.0-22m")
        device = str(getattr(configs, "device", "cuda:0" if torch.cuda.is_available() else "cpu"))
        self.model = Toto2Model.from_pretrained(model_path).to(device).eval()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.decode_block_size = int(os.environ.get("TOTO2_DECODE_BLOCK_SIZE", "768"))
        self.patch_size = int(os.environ.get("TOTO2_PATCH_SIZE", "32"))

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        centered = x_enc - means
        stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = centered / stdev

        target = x_norm.permute(0, 2, 1).contiguous()
        target_mask = torch.ones_like(target, dtype=torch.bool)
        pad_len = (-target.shape[-1]) % self.patch_size
        if pad_len:
            pad_values = torch.zeros(
                target.shape[0],
                target.shape[1],
                pad_len,
                dtype=target.dtype,
                device=target.device,
            )
            pad_mask = torch.zeros(
                target.shape[0],
                target.shape[1],
                pad_len,
                dtype=torch.bool,
                device=target.device,
            )
            target = torch.cat([pad_values, target], dim=-1)
            target_mask = torch.cat([pad_mask, target_mask], dim=-1)
        series_ids = torch.zeros(target.shape[0], target.shape[1], dtype=torch.long, device=target.device)

        with torch.no_grad():
            quantiles = self.model.forecast(
                {
                    "target": target,
                    "target_mask": target_mask,
                    "series_ids": series_ids,
                },
                horizon=self.pred_len,
                decode_block_size=self.decode_block_size,
                has_missing_values=False,
            )
        if not torch.is_tensor(quantiles):
            quantiles = torch.as_tensor(quantiles, device=target.device)
        quantiles = quantiles.to(target.device)
        median = quantiles[quantiles.shape[0] // 2]
        dec_out = median.permute(0, 2, 1).contiguous()
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == "zero_shot_forecast":
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return None
