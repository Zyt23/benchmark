import torch
import torch.nn as nn


class Model(nn.Module):
    """OmniAnomaly-style GRU-VAE adapted for QAR reconstruction scoring."""

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = int(configs.seq_len)
        self.enc_in = int(configs.enc_in)
        hidden = max(32, int(getattr(configs, "d_model", 64)))
        latent = max(8, hidden // 4)
        layers = max(1, int(getattr(configs, "e_layers", 2)))

        self.gru = nn.GRU(self.enc_in, hidden, num_layers=layers, batch_first=True)
        self.to_stats = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.PReLU(),
            nn.Linear(hidden, 2 * latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.PReLU(),
            nn.Linear(hidden, self.enc_in),
        )

    def anomaly_detection(self, x_enc):
        h, _ = self.gru(x_enc)
        stats = self.to_stats(h)
        mu, logvar = stats.chunk(2, dim=-1)
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
        else:
            z = mu
        return self.decoder(z)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name != "anomaly_detection":
            raise NotImplementedError("OmniAnomaly wrapper is only implemented for anomaly_detection")
        return self.anomaly_detection(x_enc)
