import torch
from torch import nn
import os
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from transformers import AutoModelForCausalLM

class Model(nn.Module):
    def __init__(self, configs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        model_path = os.environ.get("SUNDIAL_MODEL_PATH", "thuml/sundial-base-128m")
        device = str(getattr(configs, "device", "cuda:0" if torch.cuda.is_available() else "cpu"))
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device).eval()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        outputs = []
        for i in range(x_enc.shape[-1]):
            with torch.no_grad():
                output = self.model.generate(x_enc[...,i], max_new_tokens=self.pred_len, num_samples=20)
            output = output.mean(dim=1)
            outputs.append(output)
        dec_out = torch.stack(outputs, dim=-1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'zero_shot_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        return None
