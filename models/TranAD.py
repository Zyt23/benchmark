import torch
import torch.nn as nn


class Model(nn.Module):
    """TranAD-inspired transformer reconstruction wrapper for QAR.

    The original TranAD uses a two-phase transformer decoder.  For the shared
    QAR one-class protocol we expose a compact same-shape reconstruction model
    so it can be trained and scored by ``Exp_Anomaly_Detection``.
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
        self.pos = nn.Parameter(torch.zeros(1, self.seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=e_layers)
        self.output_proj = nn.Linear(d_model, self.enc_in)

    def anomaly_detection(self, x_enc):
        h = self.input_proj(x_enc) + self.pos[:, : x_enc.shape[1]]
        h = self.encoder(h)
        return self.output_proj(h)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name != "anomaly_detection":
            raise NotImplementedError("TranAD wrapper is only implemented for anomaly_detection")
        return self.anomaly_detection(x_enc)
