import random
from typing import Dict, Any
import torch
from torch.utils.data import Dataset

class SquadWindowDataset(Dataset):
    """
    Wraps a HF Dataset of pre-tokenized sliding-window features produced upstream.
    Each item must have:
      - input_ids, attention_mask, (token_type_ids optional)
      - answers: list of {start_token_index, end_token_index, answer_text}
    On each fetch, randomly pick ONE gold span; if none exists for this window, use -100 (ignore_index).
    """
    def __init__(self, hf_dataset, cls_token_id: int = 0):
        self.ds = hf_dataset
        self.cls_idx = int(cls_token_id)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx) -> Dict[str, Any]:
        ex = self.ds[idx]
        L = len(ex["input_ids"])

        input_ids      = torch.tensor(ex["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(ex["attention_mask"], dtype=torch.long)

        # Some backbones don't use token_type_ids; provide zeros if absent for a stable key
        tti = ex.get("token_type_ids", None)
        if tti is None:
            token_type_ids = torch.zeros(L, dtype=torch.long)
        else:
            token_type_ids = torch.tensor(tti, dtype=torch.long)

        answers = ex.get("answers", [])
        if answers:
            pick = random.randrange(len(answers))
            s = int(answers[pick]["start_token_index"])
            e = int(answers[pick]["end_token_index"])
        else:
            # IMPORTANT: ignore windows with no gold span
            s = -100
            e = -100

        start_positions = torch.tensor(s, dtype=torch.long)
        end_positions   = torch.tensor(e, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "start_positions": start_positions,
            "end_positions": end_positions,
        }

def qa_collate(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        out[k] = torch.stack([ex[k] for ex in batch], dim=0)
    return out
