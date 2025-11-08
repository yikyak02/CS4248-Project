# src/decode.py
from typing import List, Tuple, Optional

def candidate_spans(
    start_logits: List[float],
    end_logits: List[float],
    offsets: List[Optional[Tuple[int, int]]],
    topk_start: int = 20,
    topk_end: int = 20,
    max_answer_len: int = 30,
    length_penalty_alpha: float = 0.1,
) -> List[Tuple[float, int, int]]:
    """
    Build candidate spans from per-window logits and offsets.
    Returns a list of (score, start_char, end_char) sorted by descending score.
    offsets: list where question/pad tokens are None, and context tokens are (start_char, end_char).
    """
    import numpy as np

    s = np.asarray(start_logits)
    e = np.asarray(end_logits)

    # top-k indices
    start_idx = np.argsort(s)[-topk_start:][::-1]
    end_idx   = np.argsort(e)[-topk_end:][::-1]

    cands = []
    for i in start_idx:
        if offsets[i] is None:
            continue
        for j in end_idx:
            if offsets[j] is None:
                continue
            if j < i:
                continue
            length = (j - i + 1)
            if length > max_answer_len:
                continue
            st_char, _ = offsets[i]
            _, ed_char = offsets[j]
            if st_char is None or ed_char is None or ed_char <= st_char:
                continue
            score = float(s[i] + e[j] - length_penalty_alpha * length)
            cands.append((score, int(st_char), int(ed_char)))

    cands.sort(key=lambda x: x[0], reverse=True)
    return cands
