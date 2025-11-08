from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalPointerHead(nn.Module):
    """
    Conditional pointer: predict start logits, then predict end logits conditioned
    on top-K start positions via attention.
    """
    def __init__(self, hidden_size: int, topk_start: int = 5):
        super().__init__()
        self.topk = topk_start
        self.start_proj = nn.Linear(hidden_size, 1)

        self.end_q = nn.Linear(hidden_size, hidden_size)
        self.end_k = nn.Linear(hidden_size, hidden_size)
        self.end_v = nn.Linear(hidden_size, hidden_size)
        self.end_out = nn.Linear(hidden_size, 1)

        self.compare = nn.Linear(hidden_size, hidden_size)
        self._reset()

    def _reset(self):
        nn.init.xavier_uniform_(self.start_proj.weight); nn.init.zeros_(self.start_proj.bias)
        nn.init.xavier_uniform_(self.end_q.weight); nn.init.zeros_(self.end_q.bias)
        nn.init.xavier_uniform_(self.end_k.weight); nn.init.zeros_(self.end_k.bias)
        nn.init.xavier_uniform_(self.end_v.weight); nn.init.zeros_(self.end_v.bias)
        nn.init.xavier_uniform_(self.end_out.weight); nn.init.zeros_(self.end_out.bias)
        nn.init.xavier_uniform_(self.compare.weight); nn.init.zeros_(self.compare.bias)

    def forward(self, H: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, d = H.shape

        # Start logits
        start_logits = self.start_proj(H).squeeze(-1)  # [B, L]
        start_logits = start_logits.masked_fill(attention_mask == 0, -1e9)
        start_probs = F.softmax(start_logits, dim=-1)  # [B, L]

        # Top-K start positions
        K = min(self.topk, L)
        topk_probs, topk_idx = torch.topk(start_probs, K, dim=-1)  # [B, K], [B, K]

        # Gather start representations [B, K, d]
        start_repr = torch.gather(H, dim=1, index=topk_idx.unsqueeze(-1).expand(-1, -1, d))  # [B, K, d]

        # Build end logits conditioned on the K start candidates
        Q = self.end_q(start_repr)            # [B, K, d]
        K_ = self.end_k(H)                    # [B, L, d]
        V  = self.end_v(H)                    # [B, L, d]

        # attention scores: [B, K, L]
        attn_scores = torch.einsum("bkd,bld->bkl", Q, K_) / (d ** 0.5)
        attn_scores = attn_scores.masked_fill(attention_mask.unsqueeze(1) == 0, -1e9)
        attn = F.softmax(attn_scores, dim=-1)               # [B, K, L]
        cond_ctx = torch.einsum("bkl,bld->bkd", attn, V)    # [B, K, d]

        # Compare conditional context to each token to produce per-K end scores
        token_cmp = self.compare(H)                         # [B, L, d]
        end_scores_k = torch.einsum("bkd,bld->bkl", cond_ctx, token_cmp) / (d ** 0.5)  # [B, K, L]
        end_scores_k = end_scores_k.masked_fill(attention_mask.unsqueeze(1) == 0, -1e9)

        # Weighted mixture over K using normalized top-k start probabilities
        weights = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-9)  # [B, K]
        end_logits = torch.einsum("bk,bkl->bl", weights, end_scores_k)        # [B, L]
        end_logits = end_logits.masked_fill(attention_mask == 0, -1e9)

        return start_logits, end_logits

class BiaffineSpanHead(nn.Module):
    """
    Jointly scores spans (start i, end j) by scoring (i, offset=j-i) within a max length band.
    Returns span scores tensor [B, L, max_answer_len].
    """
    def __init__(self, hidden_size: int, max_answer_len: int = 30):
        super().__init__()
        self.max_answer_len = max_answer_len
        self.U = nn.Parameter(torch.empty(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.U)
        self.proj = nn.Linear(2 * hidden_size, 1)  # boundary features

    def forward(self, H: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B, L, d = H.size()
        Hu = torch.matmul(H, self.U)  # [B, L, d]
        scores = H.new_full((B, L, self.max_answer_len), fill_value=-1e9)

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
            s = s.masked_fill(m == 0, -1e9).squeeze(-1)  # [B, valid_len]
            scores[:, :valid_len, off] = s

        return scores  # [B, L, max_answer_len]
