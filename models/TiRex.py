import torch
from torch import nn
import os
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from tirex import load_model, ForecastModel


class Model(nn.Module):
    def __init__(self, configs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        model_path = os.environ.get("TIREX_MODEL_PATH", "NX-AI/TiRex")
        device = str(getattr(configs, "device", "cuda:0" if torch.cuda.is_available() else "cpu"))
        self.model = load_model(model_path, device=device, backend=os.environ.get("TIREX_BACKEND", "torch"))
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        outputs = []
        for i in range(x_enc.shape[-1]):
            quantiles, output = self.model.forecast(x_enc[...,i], prediction_length=self.pred_len)
            outputs.append(output)
        dec_out = torch.stack(outputs, dim=-1).to(x_enc.device)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'zero_shot_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        return None
