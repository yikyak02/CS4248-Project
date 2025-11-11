from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoders import HFEncoder
from .heads import ConditionalPointerHead, BiaffineSpanHead
from utils.smoothing import label_smoothing_ce

class QASpanProposer(nn.Module):
    """
    Question Answering Model combining a pre-trained encoder with a span extraction head.
    
    Supports two head types:
    - pointer: Conditional pointer network (predicts start, then end conditioned on start)
    - biaffine: Biaffine attention network (jointly scores all valid spans)
    
    Args:
        encoder_name: HuggingFace model name or path to fine-tuned checkpoint
        head_type: Type of QA head ('pointer' or 'biaffine')
        topk_start: Number of top start candidates (pointer head only)
        max_answer_len: Maximum answer length in tokens (biaffine head only)
        label_smoothing: Label smoothing factor for loss computation
        dropout: Dropout probability for regularization
    """
    def __init__(
        self, 
        encoder_name: str, 
        head_type: str = "pointer",
        topk_start: int = 5, 
        max_answer_len: int = 30,
        label_smoothing: float = 0.0, 
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.encoder = HFEncoder(encoder_name)
        self.config = self.encoder.config
        hidden = self.config.hidden_size

        self.head_type = head_type
        self.label_smoothing = float(label_smoothing)
        self.max_answer_len = int(max_answer_len)

        if head_type == "pointer":
            self.head = ConditionalPointerHead(
                hidden_size=hidden, 
                topk_start=topk_start, 
                dropout=dropout
            )
        elif head_type == "biaffine":
            self.head = BiaffineSpanHead(
                hidden_size=hidden, 
                max_answer_len=max_answer_len, 
                dropout=dropout
            )
        else:
            raise ValueError(
                f"Unknown head_type: '{head_type}'. "
                f"Must be 'pointer' or 'biaffine'."
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass through encoder and QA head.
        
        Args:
            input_ids: Token IDs [B, L]
            attention_mask: Attention mask [B, L]
            token_type_ids: Token type IDs [B, L] (optional, for BERT-style models)
            start_positions: Ground truth start positions [B] (training only)
            end_positions: Ground truth end positions [B] (training only)
            
        Returns:
            Dictionary containing:
            - For pointer head: 'start_logits' [B, L], 'end_logits' [B, L]
            - For biaffine head: 'span_scores' [B, L, max_answer_len]
            - 'loss': Scalar loss (training only)
        """
        # Encode input
        H = self.encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        )

        out: Dict[str, Any] = {}

        # === Pointer Head Path ===
        if self.head_type == "pointer":
            start_logits, end_logits = self.head(H, attention_mask)
            out["start_logits"] = start_logits
            out["end_logits"] = end_logits

            # Compute loss if training
            if start_positions is not None and end_positions is not None:
                out["loss"] = self._compute_pointer_loss(
                    start_logits, end_logits, 
                    start_positions, end_positions
                )
            return out

        # === Biaffine Head Path ===
        elif self.head_type == "biaffine":
            span_scores = self.head(H, attention_mask)
            out["span_scores"] = span_scores

            # Compute loss if training
            if start_positions is not None and end_positions is not None:
                out["loss"] = self._compute_biaffine_loss(
                    span_scores, start_positions, end_positions
                )
            return out
        
        else:
            raise ValueError(f"Unknown head_type: {self.head_type}")
    
    def _compute_pointer_loss(
        self,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        start_positions: torch.Tensor,
        end_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss for pointer head (independent start/end prediction).
        
        Args:
            start_logits: Start position logits [B, L]
            end_logits: End position logits [B, L]
            start_positions: Ground truth start positions [B]
            end_positions: Ground truth end positions [B]
            
        Returns:
            Average of start and end cross-entropy losses
        """
        if self.label_smoothing > 0.0:
            loss_s = label_smoothing_ce(
                start_logits, start_positions, 
                smoothing=self.label_smoothing, 
                ignore_index=-100, 
                reduction="mean"
            )
            loss_e = label_smoothing_ce(
                end_logits, end_positions, 
                smoothing=self.label_smoothing, 
                ignore_index=-100, 
                reduction="mean"
            )
        else:
            # Filter out examples with no valid answer (marked -100)
            valid = (start_positions != -100) & (end_positions != -100)
            if valid.any():
                loss_s = F.cross_entropy(
                    start_logits[valid], 
                    start_positions[valid], 
                    reduction="mean"
                )
                loss_e = F.cross_entropy(
                    end_logits[valid], 
                    end_positions[valid], 
                    reduction="mean"
                )
            else:
                # No valid examples in batch - return differentiable zero
                loss_s = loss_e = (start_logits.sum() + end_logits.sum()) * 0.0
        
        return 0.5 * (loss_s + loss_e)
    
    def _compute_biaffine_loss(
        self,
        span_scores: torch.Tensor,
        start_positions: torch.Tensor,
        end_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss for biaffine head (joint span prediction).
        
        The biaffine head outputs scores for (start_position, offset) pairs.
        We convert ground truth (start, end) to (start, offset) format.
        
        Args:
            span_scores: Span scores [B, L, max_answer_len]
            start_positions: Ground truth start positions [B]
            end_positions: Ground truth end positions [B]
            
        Returns:
            Cross-entropy loss over offset dimension for correct start positions
        """
        B, L, M = span_scores.shape
        
        # Filter out examples with no valid answer
        valid = (start_positions != -100) & (end_positions != -100)
        
        if valid.any():
            # Clamp positions to valid range
            sp = start_positions[valid].clamp(0, L - 1)
            ep = end_positions[valid].clamp(0, L - 1)
            
            # Convert to offset representation
            # offset=0 means single token, offset=1 means 2 tokens, etc.
            offs = (ep - sp).clamp(0, M - 1)
            
            # Extract scores for correct start positions
            rows = span_scores[valid, sp, :]  # [B_valid, M]
            
            # Cross-entropy over offset dimension
            if self.label_smoothing > 0.0:
                loss = label_smoothing_ce(
                    rows, offs, 
                    smoothing=self.label_smoothing, 
                    reduction="mean"
                )
            else:
                loss = F.cross_entropy(rows, offs, reduction="mean")
        else:
            # No valid examples - return differentiable zero
            loss = span_scores.sum() * 0.0
        
        return loss
