import torch
from torch import nn
import os
import types
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from transformers import AutoModelForCausalLM


def _patch_transformers_cache_for_sundial():
    try:
        from transformers.cache_utils import DynamicCache
    except Exception:
        return
    if not hasattr(DynamicCache, "get_max_length") and hasattr(DynamicCache, "get_seq_length"):
        DynamicCache.get_max_length = DynamicCache.get_seq_length


def _patch_sundial_generation_for_modern_transformers(model):
    """Route modern Transformers' sampling entry point to Sundial's TS decoder.

    Sundial's published checkpoint targets Transformers 4.40.1 and implements
    continuous time-series generation in ``_greedy_search``.  Transformers
    4.41+ routes both greedy and sampled decoding through ``_sample`` instead,
    which silently bypasses Sundial's multi-patch decoder and eventually builds
    invalid position ids.  Keep the repository-wide Transformers version and
    adapt only this model instance.
    """
    try:
        from packaging.version import Version
        import transformers
    except Exception:
        return
    if Version(transformers.__version__) < Version("4.41.0"):
        return

    sundial_greedy_search = model._greedy_search

    def _sample_compat(
        model_self,
        input_ids,
        logits_processor,
        stopping_criteria,
        generation_config,
        synced_gpus,
        streamer,
        **model_kwargs,
    ):
        # Transformers 4.49 eagerly supplies an empty DynamicCache.  Sundial's
        # 4.40 decoder checks only whether the object is None and therefore
        # mistakes this empty cache for a completed first decoding step.
        past_key_values = model_kwargs.get("past_key_values")
        if past_key_values is not None:
            try:
                if past_key_values.get_seq_length() == 0:
                    model_kwargs["past_key_values"] = None
            except (AttributeError, TypeError):
                pass
        return sundial_greedy_search(
            input_ids=input_ids,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            output_attentions=generation_config.output_attentions,
            output_hidden_states=generation_config.output_hidden_states,
            output_scores=generation_config.output_scores,
            output_logits=getattr(generation_config, "output_logits", None),
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )

    model._sample = types.MethodType(_sample_compat, model)

class Model(nn.Module):
    def __init__(self, configs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        _patch_transformers_cache_for_sundial()
        model_path = os.environ.get("SUNDIAL_MODEL_PATH", "thuml/sundial-base-128m")
        device = str(getattr(configs, "device", "cuda:0" if torch.cuda.is_available() else "cpu"))
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device).eval()
        _patch_sundial_generation_for_modern_transformers(self.model)
        if not hasattr(self.model, "_extract_past_from_model_output"):
            def _extract_past_from_model_output(model_self, outputs, standardize_cache_format=False):
                return getattr(outputs, "past_key_values", None)
            self.model._extract_past_from_model_output = types.MethodType(
                _extract_past_from_model_output, self.model
            )
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = int(os.environ.get("SUNDIAL_PATCH_LEN", "16"))
        self.num_samples = int(os.environ.get("SUNDIAL_NUM_SAMPLES", "1"))

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        outputs = []
        for i in range(x_enc.shape[-1]):
            series = x_enc[..., i]
            pad_len = (-series.shape[-1]) % self.patch_len
            if pad_len:
                pad = series[:, :1].repeat(1, pad_len)
                series = torch.cat([pad, series], dim=-1)
            with torch.no_grad():
                output = self.model.generate(
                    series,
                    max_new_tokens=self.pred_len,
                    num_samples=self.num_samples,
                )
            if output.ndim == 3:
                output = output.mean(dim=1)
            outputs.append(output)
        dec_out = torch.stack(outputs, dim=-1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'zero_shot_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        return None
