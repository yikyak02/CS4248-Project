from typing import Tuple
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class HFEncoder(nn.Module):
    """
    Thin wrapper around a Hugging Face encoder (no QA head).
    Returns last hidden states [B, L, d].
    """
    def __init__(self, name: str):
        super().__init__()
        self.config = AutoConfig.from_pretrained(name)
        self.backbone = AutoModel.from_pretrained(name, config=self.config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # only pass token_type_ids if supported by the model
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        if "token_type_ids" in self.backbone.forward.__code__.co_varnames:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.backbone(**kwargs)
        return outputs.last_hidden_state  # [B, L, d]
