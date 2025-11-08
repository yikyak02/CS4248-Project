# src/inspect_processed.py
"""
Quick inspector for the saved HF processed dataset (SQuAD sliding windows).
Run directly:
    python src/inspect_processed.py
"""

import os
import yaml
from collections import Counter
from datasets import DatasetDict
from transformers import AutoTokenizer

# Load config to use same tokenizer as data processing
CFG_PATH = "config/deberta_base_pointer.yaml"
if os.path.exists(CFG_PATH):
    with open(CFG_PATH, "r") as f:
        CFG = yaml.safe_load(f)
    TOKENIZER_NAME = CFG.get("encoder_name", "bert-base-uncased")
    MAX_LENGTH = int(CFG.get("max_length", 384))
else:
    print(f"Config not found at {CFG_PATH}, using defaults")
    TOKENIZER_NAME = "bert-base-uncased"
    MAX_LENGTH = 384

# -------- Defaults (edit here if needed) --------
DATASET_PATH = "data/processed_dataset"  # folder created by save_to_disk(...)
SPLIT = "train"                          # split name inside the saved dataset
SAMPLES_WITH_SPANS = 3                   # how many span-containing windows to show
# ------------------------------------------------

def summarize(ds, max_length: int):
    n = len(ds)
    with_spans = 0
    span_counts = Counter()
    bad_len = 0
    bad_mask = 0

    for r in ds:
        ans = r.get("answers", [])
        if ans:
            with_spans += 1
            span_counts[len(ans)] += 1
        if len(r["input_ids"]) != max_length:
            bad_len += 1
        if not set(r["attention_mask"]).issubset({0, 1}):
            bad_mask += 1

    print("\n=== Dataset Summary ===")
    print(f"total windows              : {n}")
    print(f"windows with >=1 gold span : {with_spans} ({(with_spans / max(n,1)) * 100:.2f}%)")
    print(f"windows with 0 spans       : {n - with_spans} ({((n - with_spans) / max(n,1)) * 100:.2f}%)")
    total_spans = sum(k * v for k, v in span_counts.items())
    print(f"total span annotations     : {total_spans}")
    if span_counts:
        top = span_counts.most_common(5)
        print(f"span-count per window (top): {top}")
    print(f"non-{max_length} token rows: {bad_len}")
    print(f"non-binary attention masks : {bad_mask}")

def list_ids(ds, top_k: int = 10):
    id_counts = Counter(r["id"] for r in ds)
    print("\n=== Question-ID frequency (top) ===")
    for qid, cnt in id_counts.most_common(top_k):
        print(f"{qid}: {cnt} windows")
    print(f"unique question ids: {len(id_counts)}")

def show_samples(ds, tokenizer, samples: int, show_empty: bool = True):
    print("\n=== Sample windows WITH spans ===")
    shown = 0
    for r in ds:
        if not r["answers"]:
            continue
        s = r["answers"][0]["start_token_index"]
        e = r["answers"][0]["end_token_index"]
        span_tokens = tokenizer.convert_ids_to_tokens(r["input_ids"][s:e + 1])
        span_text = tokenizer.convert_tokens_to_string(span_tokens)
        print(f"- id: {r['id']}")
        print(f"  spans_in_window: {len(r['answers'])}")
        print(f"  first_span_token_idx: ({s}, {e})")
        preview = span_tokens[:12]
        print(f"  first_span_tokens: {preview}{' ...' if len(span_tokens) > 12 else ''}")
        print(f"  first_span_decoded: \"{span_text}\"")
        first20 = tokenizer.convert_ids_to_tokens(r["input_ids"][:20])
        print(f"  first_20_tokens: {first20}")
        shown += 1
        if shown >= samples:
            break

    if show_empty:
        print("\n=== Sample windows WITHOUT spans ===")
        shown = 0
        for r in ds:
            if r["answers"]:
                continue
            print(f"- id: {r['id']} (answers=[])")
            first20 = tokenizer.convert_ids_to_tokens(r["input_ids"][:20])
            print(f"  first_20_tokens: {first20}")
            shown += 1
            if shown >= max(1, samples // 2):
                break

def main():
    try:
        print(f"Loading dataset from: {DATASET_PATH}")
        print(f"Using tokenizer: {TOKENIZER_NAME}")
        print(f"Expected max length: {MAX_LENGTH}")
        
        ds = DatasetDict.load_from_disk(DATASET_PATH)[SPLIT]
        print("\n=== Loaded dataset ===")
        print(ds)
        print("\n=== Features ===")
        print(ds.features)

        summarize(ds, MAX_LENGTH)
        list_ids(ds, top_k=10)

        tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
        show_samples(ds, tok, samples=SAMPLES_WITH_SPANS, show_empty=True)
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure the processed dataset exists at:", DATASET_PATH)
        print("2. Run data processing first: python src/data_processing.py")
        print("3. Check if config file exists at:", CFG_PATH)

if __name__ == "__main__":
    main()
