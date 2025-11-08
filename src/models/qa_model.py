from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoders import HFEncoder
from .heads import ConditionalPointerHead, BiaffineSpanHead
from utils.smoothing import label_smoothing_ce

class QASpanProposer(nn.Module):
    """
    Encoder + span-aware head (pointer or biaffine).
    Training: returns loss (+ logits/scores).
    Inference: returns logits/scores only.
    """
    def __init__(self, encoder_name: str, head_type: str = "pointer",
                 topk_start: int = 5, max_answer_len: int = 30,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.encoder = HFEncoder(encoder_name)
        self.config = self.encoder.config
        hidden = self.config.hidden_size

        self.head_type = head_type
        self.label_smoothing = float(label_smoothing)
        self.max_answer_len = int(max_answer_len)

        if head_type == "pointer":
            self.head = ConditionalPointerHead(hidden_size=hidden, topk_start=topk_start)
        elif head_type == "biaffine":
            self.head = BiaffineSpanHead(hidden_size=hidden, max_answer_len=max_answer_len)
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        H = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        out: Dict[str, Any] = {}

        if self.head_type == "pointer":
            start_logits, end_logits = self.head(H, attention_mask)  # [B, L], [B, L]
            out["start_logits"] = start_logits
            out["end_logits"] = end_logits

            if start_positions is not None and end_positions is not None:
                # DO NOT clamp; keep -100 to be ignored
                if self.label_smoothing > 0.0:
                    ls = self.label_smoothing
                    loss_s = label_smoothing_ce(
                        start_logits, start_positions, smoothing=ls, ignore_index=-100, reduction="mean"
                    )
                    loss_e = label_smoothing_ce(
                        end_logits,   end_positions,   smoothing=ls, ignore_index=-100, reduction="mean"
                    )
                else:
                    loss_s = F.cross_entropy(start_logits, start_positions, ignore_index=-100, reduction="mean")
                    loss_e = F.cross_entropy(end_logits,   end_positions,   ignore_index=-100, reduction="mean")
                out["loss"] = 0.5 * (loss_s + loss_e)
            return out

        # ----- biaffine path -----
        span_scores = self.head(H, attention_mask)   # [B, L, max_len]
        out["span_scores"] = span_scores

        if start_positions is not None and end_positions is not None:
            B, L, M = span_scores.shape
            valid = (start_positions != -100) & (end_positions != -100)
            if valid.any():
                sp = start_positions[valid].clamp(0, L - 1)
                ep = end_positions[valid].clamp(0, L - 1)
                offs = (ep - sp).clamp(0, M - 1)                # [B_valid]
                rows = span_scores[valid, sp, :]                 # [B_valid, M]
                if self.label_smoothing > 0.0:
                    loss = label_smoothing_ce(rows, offs, smoothing=self.label_smoothing, reduction="mean")
                else:
                    loss = F.cross_entropy(rows, offs, reduction="mean")
            else:
                loss = span_scores.sum() * 0.0  # differentiable zero
            out["loss"] = loss

        return out
