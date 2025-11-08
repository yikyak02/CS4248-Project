import torch
import torch.nn.functional as F

def label_smoothing_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    smoothing: float = 0.1,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Cross-entropy with label smoothing that supports ignore_index and reduction.
    Args:
      logits: [B, C]
      target: [B] (class indices) or ignore_index for samples to skip
      smoothing: float in [0, 1)
      ignore_index: index value in target to ignore
      reduction: "mean" | "sum" | "none"
    """
    assert logits.dim() == 2, "logits must be [B, C]"
    B, C = logits.shape

    # mask out ignored rows
    valid = target != ignore_index
    if not valid.any():
        # no valid rows in this batch; return differentiable zero
        return logits.sum() * 0.0

    logits_v = logits[valid]            # [Bv, C]
    target_v = target[valid].long()     # [Bv]

    with torch.no_grad():
        true_dist = torch.zeros_like(logits_v).scatter_(1, target_v.unsqueeze(1), 1.0)
        true_dist = true_dist * (1.0 - smoothing) + smoothing / C

    log_probs = F.log_softmax(logits_v, dim=-1)
    loss_vec = -(true_dist * log_probs).sum(dim=-1)  # [Bv]

    if reduction == "mean":
        return loss_vec.mean()
    if reduction == "sum":
        return loss_vec.sum()
    return loss_vec  # "none"
