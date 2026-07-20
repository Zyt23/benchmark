"""TiRex-2 zero-shot multivariate forecasting adapter.

TiRex-2 expects each item as ``(n_targets, context_length)`` and returns nine
quantiles with shape ``(n_targets, 9, prediction_length)``.  The TSLib QAR
loader uses ``(batch, context_length, n_targets)``, so this adapter only changes
layout and returns the median forecast.  No QAR sample is used to fit TiRex-2.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        try:
            from tirex2 import TimeseriesType, load_model
        except ImportError as exc:
            raise ImportError(
                "TiRex-2 requires the optional `tirex-2` package in a Python "
                "3.11+ environment with its official PyTorch dependencies."
            ) from exc

        self._timeseries_type = TimeseriesType
        self.task_name = configs.task_name
        self.pred_len = int(configs.pred_len)
        model_path = os.environ.get("TIREX2_MODEL_PATH", "NX-AI/TiRex-2")
        device = os.environ.get(
            "TIREX2_DEVICE",
            "cuda" if torch.cuda.is_available() else "cpu",
        )
        self.model = load_model(model_path, device=device)

    def forecast(self, x_enc):
        # B,L,C -> list[(C,L)].  TiRex-2 handles all QAR variables jointly.
        contexts = x_enc.detach().float().cpu().permute(0, 2, 1).numpy()
        series = [
            self._timeseries_type(
                target=torch.from_numpy(np.asarray(context, dtype=np.float32)),
                past_covariates=None,
                future_covariates=None,
            )
            for context in contexts
        ]
        forecasts = self.model.forecast(
            timeseries=series,
            prediction_length=self.pred_len,
            output_type="numpy",
        )
        medians = []
        for forecast in forecasts:
            forecast = np.asarray(forecast)
            if forecast.ndim != 3 or forecast.shape[1] < 5:
                raise ValueError(
                    "Unexpected TiRex-2 output shape {}; expected (C,9,H)".format(
                        forecast.shape
                    )
                )
            medians.append(forecast[:, 4, :].T)
        return torch.as_tensor(
            np.stack(medians, axis=0),
            dtype=x_enc.dtype,
            device=x_enc.device,
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == "zero_shot_forecast":
            return self.forecast(x_enc)
        return None
