import torch
import torch.nn as nn


class Model(nn.Module):
    """USAD-style autoencoder adapted to the QAR anomaly-detection interface.

    The PRSOV copy on the server expects ``x_enc + exog_future + endo_future``.
    QAR one-class anomaly detection only passes one compact window ``x_enc`` and
    scores reconstruction error, so this wrapper reconstructs the whole window
    and returns a tensor with the same shape as the input.
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = int(configs.seq_len)
        self.enc_in = int(configs.enc_in)
        n_flat = self.seq_len * self.enc_in
        hidden = max(32, int(getattr(configs, "d_model", 64)))
        latent = max(8, hidden // 4)

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flat, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, latent),
            nn.ReLU(True),
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, n_flat),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, n_flat),
        )

    def anomaly_detection(self, x_enc):
        z = self.encoder(x_enc)
        recon1 = self.decoder1(z)
        # Keep the second decoder in the graph during training, matching the
        # spirit of USAD, while returning the primary reconstruction used by the
        # shared QAR scoring code.
        _ = self.decoder2(z)
        return recon1.view(x_enc.shape[0], self.seq_len, self.enc_in)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name != "anomaly_detection":
            raise NotImplementedError("USAD wrapper is only implemented for anomaly_detection")
        return self.anomaly_detection(x_enc)
