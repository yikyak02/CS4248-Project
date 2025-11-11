"""
Question Answering Prediction Script.

Generates predictions for SQuAD-format dev/test data using a trained QA model.
Supports both pointer and biaffine heads with sliding window processing.
"""
import json
import os
import argparse
from typing import List, Dict, Any, Optional, Tuple
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from models.qa_model import QASpanProposer
from decode import candidate_spans


def decode_biaffine_window(
    span_scores: torch.Tensor,
    offsets: List[Optional[Tuple[int, int]]],
    context: str,
    topk: int = 20,
    max_answer_len: int = 30,
    length_penalty_alpha: float = 0.1
) -> List[Tuple[float, int, int]]:
    """
    Decode biaffine span scores for a single window to extract top answer candidates.
    
    The biaffine head outputs scores for (start_position, offset) pairs where:
    - start_position: Token index where span begins
    - offset: Span length minus 1 (offset=0 → 1 token, offset=1 → 2 tokens)
    
    This function:
    1. Flattens the [L, M] score tensor
    2. Gets top-K highest scoring (start, offset) pairs
    3. Filters out question/padding tokens
    4. Converts to character-level spans in original context
    
    Args:
        span_scores: Span scores [L, M] where scores[i, j] = score for span
                     starting at position i with offset j
        offsets: Token-to-character mappings. offsets[i] = (start_char, end_char)
                for context tokens, None for question/padding tokens
        context: Original context string
        topk: Number of valid candidates to return
        max_answer_len: Maximum answer length in tokens (should match training)
        length_penalty_alpha: Penalty coefficient for longer spans (reduces score)
    
    Returns:
        List of (penalized_score, start_char, end_char) tuples, sorted by score.
        Character positions refer to the original context string.
    """
    L, M = span_scores.shape
    candidates = []
    
    # Flatten scores and get top-k (start_pos, offset) pairs
    # Increase topk to account for filtering out question/padding tokens
    flat_scores = span_scores.flatten()
    topk_actual = min(topk * 10, L * M)  # 10x buffer for filtering
    topk_values, topk_indices = torch.topk(flat_scores, topk_actual)
    
    for score, flat_idx in zip(topk_values, topk_indices):
        # Convert flat index back to (start_pos, offset)
        start_pos = int(flat_idx // M)
        offset = int(flat_idx % M)
        end_pos = start_pos + offset
        
        # Skip if positions are out of bounds
        if start_pos >= L or end_pos >= L:
            continue
        
        # Get character offsets
        start_offset = offsets[start_pos]
        end_offset = offsets[end_pos]
        
        # Skip if either is None (question/padding token)
        if start_offset is None or end_offset is None:
            continue
        
        start_char, _ = start_offset
        _, end_char = end_offset
        
        # Skip invalid character spans
        if start_char is None or end_char is None or end_char <= start_char:
            continue
        
        # Check token length constraint
        token_length = end_pos - start_pos + 1
        if token_length > max_answer_len:
            continue
        
        # Apply length penalty (same as pointer head for consistency)
        penalized_score = float(score) - length_penalty_alpha * token_length
        
        candidates.append((penalized_score, int(start_char), int(end_char)))
        
        # Early exit once we have enough good candidates
        if len(candidates) >= topk:
            break
    
    return candidates


def read_squad_dev(path: str) -> List[Dict[str, Any]]:
    """
    Read SQuAD-format JSON and extract question-context pairs.
    
    Args:
        path: Path to SQuAD JSON file
        
    Returns:
        List of dicts with keys: 'id', 'question', 'context'
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)["data"]
    
    examples = []
    for article in data:
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                examples.append({
                    "id": qa["id"], 
                    "question": qa["question"], 
                    "context": context
                })
    return examples


def build_windows(
    tokenizer, 
    question: str, 
    context: str, 
    max_length: int, 
    stride: int
) -> List[Dict[str, Any]]:
    """
    Tokenize question-context pair into sliding windows.
    
    For long contexts, creates multiple overlapping windows to ensure
    all parts of the context are covered.
    
    Args:
        tokenizer: HuggingFace tokenizer
        question: Question string
        context: Context string
        max_length: Maximum tokens per window (including question)
        stride: Stride for sliding windows (overlap = max_length - stride)
        
    Returns:
        List of feature dicts, one per window, containing:
        - input_ids: Token IDs [L]
        - attention_mask: Attention mask [L]
        - token_type_ids: Token type IDs [L] (if applicable)
        - offsets: Token-to-character mappings (None for question tokens)
    """
    enc = tokenizer(
        question,
        context,
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
        max_length=max_length,
        stride=stride,
        truncation="only_second",  # Only truncate context, not question
        padding="max_length",
    )
    
    feats = []
    for i in range(len(enc["input_ids"])):
        seq_ids = enc.sequence_ids(i)
        offsets = enc["offset_mapping"][i]
        
        # Keep only context token offsets (question/padding → None)
        ctx_offsets = [
            o if seq_ids[tok_i] == 1 else None 
            for tok_i, o in enumerate(offsets)
        ]
        
        feats.append({
            "input_ids": torch.tensor(enc["input_ids"][i], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"][i], dtype=torch.long),
            "token_type_ids": torch.tensor(
                enc.get("token_type_ids", [[0]*len(enc["input_ids"][i])])[i], 
                dtype=torch.long
            ) if "token_type_ids" in enc else None,
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

    # Try to load QA head metadata if available
    head_type = "pointer"
    topk_start_meta = 5
    max_answer_len_meta = args.max_answer_len
    cfg_path = os.path.join(args.checkpoint_dir, "qa_config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r") as f:
                meta = json.load(f)
            head_type = meta.get("head_type", head_type)
            topk_start_meta = int(meta.get("topk_start", topk_start_meta))
            max_answer_len_meta = int(meta.get("max_answer_len", max_answer_len_meta))
        except Exception:
            pass

    # Load tokenizer and model FROM CHECKPOINT DIR (so encoder weights are the fine-tuned ones)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, use_fast=True)
    model = QASpanProposer(
        encoder_name=args.checkpoint_dir,  # load fine-tuned encoder from folder
        head_type=head_type,
        topk_start=topk_start_meta,
        max_answer_len=max_answer_len_meta,
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

            all_cands = []  # Collect candidates from all windows

            # Process windows in batches
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
                
                # Handle different head types
                if "span_scores" in out:
                    # Biaffine head: decode each window separately
                    span_scores = out["span_scores"]  # [B, L, M]
                    for b_idx in range(span_scores.size(0)):
                        window_scores = span_scores[b_idx]  # [L, M]
                        offsets = batch_feats[b_idx]["offsets"]
                        
                        # Decode biaffine scores for this window
                        window_cands = decode_biaffine_window(
                            window_scores.cpu(),
                            offsets,
                            ex["context"],
                            topk=args.topk_start,
                            max_answer_len=args.max_answer_len,
                            length_penalty_alpha=args.length_penalty_alpha
                        )
                        all_cands.extend(window_cands)
                else:
                    # Pointer head: use existing decode logic
                    start_logits = out["start_logits"].cpu().tolist()
                    end_logits = out["end_logits"].cpu().tolist()
                    
                    for s_log, e_log, offs in zip(start_logits, end_logits, 
                                                   [f["offsets"] for f in batch_feats]):
                        cands = candidate_spans(
                            s_log, e_log, offs,
                            topk_start=args.topk_start,
                            topk_end=args.topk_end,
                            max_answer_len=args.max_answer_len,
                            length_penalty_alpha=args.length_penalty_alpha
                        )
                        all_cands.extend(cands)

            # Select best candidate across all windows for this question
            if not all_cands:
                preds[ex["id"]] = ""
            else:
                best = max(all_cands, key=lambda x: x[0])
                st_char, ed_char = best[1], best[2]
                preds[ex["id"]] = ex["context"][st_char:ed_char].strip()

    # Save predictions
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False)
    print(f"Wrote predictions to {args.output}")


if __name__ == "__main__":
    main()
