import torch
import torch.nn as nn


class Model(nn.Module):
    """Anomaly-Transformer-style self-attention reconstruction wrapper.

    This is intentionally shaped like the TSLib anomaly interface: input and
    output are both ``[batch, seq_len, channels]``.  The thresholding and
    TN/FP/FN/TP computation are handled by ``Exp_Anomaly_Detection``.
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = int(configs.seq_len)
        self.enc_in = int(configs.enc_in)
        d_model = max(16, int(getattr(configs, "d_model", 64)))
        n_heads = int(getattr(configs, "n_heads", 4))
        while d_model % n_heads != 0 and n_heads > 1:
            n_heads -= 1
        e_layers = max(1, int(getattr(configs, "e_layers", 2)))
        d_ff = max(d_model, int(getattr(configs, "d_ff", 128)))
        dropout = float(getattr(configs, "dropout", 0.1))

        self.input_proj = nn.Linear(self.enc_in, d_model)
        self.position = nn.Parameter(torch.zeros(1, self.seq_len, d_model))
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(e_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, self.enc_in)

    def anomaly_detection(self, x_enc):
        h = self.input_proj(x_enc) + self.position[:, : x_enc.shape[1]]
        for layer in self.layers:
            h = layer(h)
        return self.output_proj(self.norm(h))

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name != "anomaly_detection":
            raise NotImplementedError("AnomalyTransformer wrapper is only implemented for anomaly_detection")
        return self.anomaly_detection(x_enc)
