"""
Question Answering Head Implementations.

This module implements different architectures for extracting answer spans
from contextualized token representations.
"""
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# Mask value for padding tokens (safe for mixed precision training)
MASK_VAL = -1e2


class ConditionalPointerHead(nn.Module):
    """
    Conditional Pointer Network for span extraction.
    
    This head predicts answer spans in two stages:
    1. Predict start position independently
    2. Predict end position conditioned on top-K start candidates via attention
    
    The conditional mechanism allows the model to learn which end positions
    are likely given a particular start position, improving span coherence.
    
    Architecture:
    - Start: Linear projection of hidden states
    - End: Attention-based conditioning on top-K start positions
    - Uses 2x hidden dimension for improved expressiveness
    
    Args:
        hidden_size: Dimension of encoder hidden states
        topk_start: Number of top start candidates to condition on
        dropout: Dropout probability for regularization
    """
    def __init__(self, hidden_size: int, topk_start: int = 10, dropout: float = 0.1):
        super().__init__()
        self.topk = topk_start
        self.dropout = nn.Dropout(dropout)
        
        # Expand to 2x dimensions for better expressiveness
        self.expanded_dim = hidden_size * 2
        
        # Start prediction (simple projection)
        self.start_proj = nn.Linear(hidden_size, 1)

        # End prediction with attention (expanded dimensions)
        self.end_q = nn.Linear(hidden_size, self.expanded_dim)
        self.end_k = nn.Linear(hidden_size, self.expanded_dim)
        self.end_v = nn.Linear(hidden_size, self.expanded_dim)

        # Comparison layer for scoring end positions
        self.compare = nn.Linear(hidden_size, self.expanded_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.start_proj.weight)
        nn.init.zeros_(self.start_proj.bias)
        nn.init.xavier_uniform_(self.end_q.weight)
        nn.init.zeros_(self.end_q.bias)
        nn.init.xavier_uniform_(self.end_k.weight)
        nn.init.zeros_(self.end_k.bias)
        nn.init.xavier_uniform_(self.end_v.weight)
        nn.init.zeros_(self.end_v.bias)
        nn.init.xavier_uniform_(self.compare.weight)
        nn.init.zeros_(self.compare.bias)

    def forward(
        self, 
        H: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to compute start and end logits.
        
        Args:
            H: Encoder hidden states [B, L, hidden_size]
            attention_mask: Attention mask [B, L] (1 = valid, 0 = padding)
            
        Returns:
            start_logits: Start position scores [B, L]
            end_logits: End position scores [B, L]
        """
        B, L, d = H.shape
        
        # Apply dropout to encoder outputs
        H = self.dropout(H)

        # Start logits
        start_logits = self.start_proj(H).squeeze(-1)  # [B, L]
        start_logits = start_logits.masked_fill(attention_mask == 0, MASK_VAL)
        start_probs = F.softmax(start_logits, dim=-1)  # [B, L]

        # Top-K start positions
        K = min(self.topk, L)
        topk_probs, topk_idx = torch.topk(start_probs, K, dim=-1)  # [B, K], [B, K]

        # Gather start representations [B, K, d]
        start_repr = torch.gather(H, dim=1, index=topk_idx.unsqueeze(-1).expand(-1, -1, d))  # [B, K, d]

        # Build end logits conditioned on the K start candidates (using expanded dims)
        Q = self.end_q(start_repr)            # [B, K, expanded_dim]
        K_ = self.end_k(H)                    # [B, L, expanded_dim]
        V  = self.end_v(H)                    # [B, L, expanded_dim]

        # Attention scores: [B, K, L]
        attn_scores = torch.einsum("bkd,bld->bkl", Q, K_) / (self.expanded_dim ** 0.5)
        attn_scores = attn_scores.masked_fill(attention_mask.unsqueeze(1) == 0, MASK_VAL)
        attn = F.softmax(attn_scores, dim=-1)               # [B, K, L]
        cond_ctx = torch.einsum("bkl,bld->bkd", attn, V)    # [B, K, expanded_dim]

        # Compare conditional context to each token to produce per-K end scores
        token_cmp = self.compare(H)                         # [B, L, expanded_dim]
        end_scores_k = torch.einsum("bkd,bld->bkl", cond_ctx, token_cmp) / (self.expanded_dim ** 0.5)  # [B, K, L]
        end_scores_k = end_scores_k.masked_fill(attention_mask.unsqueeze(1) == 0, MASK_VAL)

        # Weighted mixture over K using normalized top-k start probabilities
        weights = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-9)  # [B, K]
        end_logits = torch.einsum("bk,bkl->bl", weights, end_scores_k)        # [B, L]
        end_logits = end_logits.masked_fill(attention_mask == 0, MASK_VAL)

        return start_logits, end_logits

class BiaffineSpanHead(nn.Module):
    """
    Biaffine Attention Head for joint span scoring.
    
    This head jointly scores all valid (start, offset) pairs using biaffine attention,
    where offset represents the span length minus 1 (offset=0 → 1 token, offset=1 → 2 tokens).
    
    The scoring combines:
    1. Biaffine interaction: start^T · U · end (captures token compatibility)
    2. Boundary features: [start; end] concatenation (captures boundary patterns)
    
    This approach is more efficient than scoring all L^2 (start, end) pairs by
    restricting to valid spans within max_answer_len.
    
    Args:
        hidden_size: Dimension of encoder hidden states
        max_answer_len: Maximum answer length in tokens (limits offset range)
        dropout: Dropout probability for regularization
    """
    def __init__(self, hidden_size: int, max_answer_len: int = 30, dropout: float = 0.1):
        super().__init__()
        self.max_answer_len = max_answer_len
        self.dropout = nn.Dropout(dropout)
        
        # Biaffine parameter matrix
        self.U = nn.Parameter(torch.empty(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.U)
        
        # Boundary feature projection
        self.proj = nn.Linear(2 * hidden_size, 1)

    def forward(self, H: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute span scores.
        
        Args:
            H: Encoder hidden states [B, L, hidden_size]
            attention_mask: Attention mask [B, L] (1 = valid, 0 = padding)
            
        Returns:
            span_scores: Span scores [B, L, max_answer_len]
                         scores[b, i, j] = score for span starting at position i with offset j
        """
        B, L, d = H.size()
        
        # Apply dropout
        H = self.dropout(H)
        
        Hu = torch.matmul(H, self.U)  # [B, L, d]
        scores = H.new_full((B, L, self.max_answer_len), fill_value=MASK_VAL)

        for off in range(self.max_answer_len):
            valid_len = L - off
            if valid_len <= 0:
                break
            start_Hu = Hu[:, :valid_len, :]        # [B, valid_len, d]
            end_H   = H[:, off:, :]                # [B, valid_len, d]
            bil = (start_Hu * end_H).sum(-1, keepdim=True)  # [B, valid_len, 1]
            fea = torch.cat([H[:, :valid_len, :], H[:, off:, :]], dim=-1)  # [B, valid_len, 2d]
            s = bil + self.proj(fea)                                   # [B, valid_len, 1]
            m = (attention_mask[:, :valid_len] * attention_mask[:, off:]).unsqueeze(-1)  # [B, valid_len, 1]
            s = s.masked_fill(m == 0, MASK_VAL).squeeze(-1)  # [B, valid_len]
            scores[:, :valid_len, off] = s

        return scores  # [B, L, max_answer_len]
