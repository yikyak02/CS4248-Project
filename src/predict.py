# src/predict.py
import json
import argparse
from typing import List, Dict, Any, Optional
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from models.qa_model import QASpanProposer
from decode import candidate_spans

def read_squad_dev(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)["data"]
    examples = []
    for article in data:
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                examples.append({"id": qa["id"], "question": qa["question"], "context": context})
    return examples

def build_windows(tokenizer, question: str, context: str, max_length: int, stride: int):
    enc = tokenizer(
        question,
        context,
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
        max_length=max_length,
        stride=stride,
        truncation="only_second",
        padding="max_length",
    )
    feats = []
    for i in range(len(enc["input_ids"])):
        seq_ids = enc.sequence_ids(i)
        offsets = enc["offset_mapping"][i]
        # keep only context offsets; question/pad -> None
        ctx_offsets = [o if seq_ids[tok_i] == 1 else None for tok_i, o in enumerate(offsets)]
        feats.append({
            "input_ids": torch.tensor(enc["input_ids"][i], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"][i], dtype=torch.long),
            "token_type_ids": torch.tensor(enc.get("token_type_ids", [[0]*len(enc["input_ids"][i])])[i], dtype=torch.long)
                if "token_type_ids" in enc else None,
            "offsets": ctx_offsets
        })
    return feats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", type=str, required=True,
                    help="Folder that contains fine-tuned encoder (save_pretrained) and qa_head.pt")
    ap.add_argument("--dev_json", type=str, required=True)
    ap.add_argument("--output", type=str, default="predictions.json")
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--topk_start", type=int, default=20)
    ap.add_argument("--topk_end", type=int, default=20)
    ap.add_argument("--max_answer_len", type=int, default=30)
    ap.add_argument("--length_penalty_alpha", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model FROM CHECKPOINT DIR (so encoder weights are the fine-tuned ones)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, use_fast=True)
    model = QASpanProposer(
        encoder_name=args.checkpoint_dir,  # load fine-tuned encoder from folder
        head_type="pointer",
        topk_start=5  # this only affects how end logits are formed; training used same
    ).to(device)
    # load head weights
    head_state = torch.load(f"{args.checkpoint_dir.rstrip('/')}/qa_head.pt", map_location="cpu")
    model.load_state_dict(head_state, strict=False)
    model.eval()

    examples = read_squad_dev(args.dev_json)
    preds: Dict[str, str] = {}

    with torch.no_grad():
        for ex in tqdm(examples, desc="Predict"):
            feats = build_windows(
                tokenizer, ex["question"], ex["context"], args.max_length, args.stride
            )

            # Run in small batches for speed
            start_logits_all: List[List[float]] = []
            end_logits_all:   List[List[float]] = []
            offsets_all = []

            # simple batching
            for i in range(0, len(feats), args.batch_size):
                batch_feats = feats[i:i+args.batch_size]
                input_ids = torch.stack([f["input_ids"] for f in batch_feats]).to(device)
                attention_mask = torch.stack([f["attention_mask"] for f in batch_feats]).to(device)
                if batch_feats[0]["token_type_ids"] is not None:
                    token_type_ids = torch.stack([f["token_type_ids"] for f in batch_feats]).to(device)
                else:
                    token_type_ids = None

                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                start_logits = out["start_logits"].cpu().tolist()
                end_logits   = out["end_logits"].cpu().tolist()

                start_logits_all.extend(start_logits)
                end_logits_all.extend(end_logits)
                offsets_all.extend([f["offsets"] for f in batch_feats])

            # Build candidates across all windows and pick best
            all_cands = []
            for s_log, e_log, offs in zip(start_logits_all, end_logits_all, offsets_all):
                cands = candidate_spans(
                    s_log, e_log, offs,
                    topk_start=args.topk_start,
                    topk_end=args.topk_end,
                    max_answer_len=args.max_answer_len,
                    length_penalty_alpha=args.length_penalty_alpha
                )
                all_cands.extend(cands)

            if not all_cands:
                preds[ex["id"]] = ""  # fallback (should be rare on SQuAD v1.1)
            else:
                best = max(all_cands, key=lambda x: x[0])
                st_char, ed_char = best[1], best[2]
                preds[ex["id"]] = ex["context"][st_char:ed_char]

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False)
    print(f"Wrote predictions to {args.output}")

if __name__ == "__main__":
    main()
