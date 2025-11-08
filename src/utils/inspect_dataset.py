# src/inspect_dataset.py
"""
Inspect the saved HF dataset produced by data_processing.py.
Runs with no flags; reads config/deberta_base_pointer.yaml for settings.

Usage:
    python -m src.inspect_dataset
"""

import os
from collections import Counter

import yaml
from datasets import DatasetDict
from transformers import AutoTokenizer


# ---------- Defaults (no CLI) ----------
CFG_PATH = "config/deberta_base_pointer.yaml"
DATASET_PATH = "data/processed_dataset"   # unchanged output path
SPLIT = "train"
SAMPLES_WITH_SPANS = 3
# ---------------------------------------


def load_cfg(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # pull what we need (with safe fallbacks)
    enc = cfg.get("encoder_name", "microsoft/deberta-v3-base")
    max_len = int(cfg.get("max_length", 384))
    return enc, max_len


def summarize(ds, max_length: int):
    n = len(ds)
    with_spans = 0
    span_counts = Counter()
    bad_len = 0
    bad_mask = 0
    missing_seg = 0

    for r in ds:
        ans = r.get("answers", [])
        if ans:
            with_spans += 1
            span_counts[len(ans)] += 1

        if len(r["input_ids"]) != max_length:
            bad_len += 1

        # attention_mask should be 0/1
        if not set(r["attention_mask"]).issubset({0, 1}):
            bad_mask += 1

        if "token_type_ids" not in r:
            missing_seg += 1

    print("\n=== Dataset Summary ===")
    print(f"total windows                 : {n}")
    print(f"windows with ≥1 gold span     : {with_spans} ({with_spans/max(1,n)*100:.2f}%)")
    print(f"windows with 0 spans          : {n-with_spans} ({(n-with_spans)/max(1,n)*100:.2f}%)")
    total_spans = sum(k*v for k, v in span_counts.items())
    print(f"total span annotations        : {total_spans}")
    if span_counts:
        print(f"span-count per window (top 5) : {span_counts.most_common(5)}")
    print(f"rows with length != {max_length}: {bad_len}")
    print(f"non-binary attention masks    : {bad_mask}")
    print(f"rows missing token_type_ids   : {missing_seg}  (expected for DeBERTa/Roberta)")


def top_ids(ds, k: int = 10):
    id_counts = Counter(r["id"] for r in ds)
    print("\n=== Question-ID frequency (top) ===")
    for qid, cnt in id_counts.most_common(k):
        print(f"{qid}: {cnt} windows")
    print(f"unique question ids: {len(id_counts)}")


def show_samples(ds, tokenizer, samples: int):
    print("\n=== Sample windows WITH spans ===")
    shown = 0
    for r in ds:
        if not r["answers"]:
            continue
        s = r["answers"][0]["start_token_index"]
        e = r["answers"][0]["end_token_index"]
        span_tokens = tokenizer.convert_ids_to_tokens(r["input_ids"][s:e+1])
        span_text = tokenizer.convert_tokens_to_string(span_tokens)
        print(f"- id: {r['id']}")
        print(f"  spans_in_window      : {len(r['answers'])}")
        print(f"  first_span_token_idx : ({s}, {e})")
        preview = span_tokens[:12]
        print(f"  first_span_tokens    : {preview}{' ...' if len(span_tokens)>12 else ''}")
        print(f"  first_span_decoded   : \"{span_text}\"")
        first20 = tokenizer.convert_ids_to_tokens(r["input_ids"][:20])
        print(f"  first_20_tokens      : {first20}")
        shown += 1
        if shown >= samples:
            break

    print("\n=== Sample windows WITHOUT spans ===")
    shown = 0
    for r in ds:
        if r["answers"]:
            continue
        print(f"- id: {r['id']} (answers=[])")
        first20 = tokenizer.convert_ids_to_tokens(r["input_ids"][:20])
        print(f"  first_20_tokens      : {first20}")
        shown += 1
        if shown >= max(1, samples // 2):
            break


def main():
    encoder_name, max_len = load_cfg(CFG_PATH)
    ds = DatasetDict.load_from_disk(DATASET_PATH)[SPLIT]

    print("=== Loaded dataset ===")
    print(ds)
    print("\n=== Features ===")
    print(ds.features)

    summarize(ds, max_len)
    top_ids(ds, k=10)

    tok = AutoTokenizer.from_pretrained(encoder_name, use_fast=True)
    show_samples(ds, tok, samples=SAMPLES_WITH_SPANS)

    # Quick integrity check on token_type_ids shape if present
    has_seg = "token_type_ids" in ds.column_names
    if has_seg:
        bad_seg = sum(1 for r in ds if len(r["token_type_ids"]) != max_len)
        if bad_seg:
            print(f"\n[Warn] {bad_seg} rows have token_type_ids length != {max_len}")
    print("\nInspection done.")


if __name__ == "__main__":
    main()
