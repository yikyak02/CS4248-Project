CS4248 Project — Extractive QA with Span‑Aware Proposer

Abstract
- Build an extractive QA model that predicts answer spans from passages on SQuAD v1.1. Improve over the vanilla “independent start/end” baseline via a stronger encoder (DeBERTa‑v3‑small), span‑aware heads (conditional pointer or biaffine), and stable training (linear warmup, gradient clipping, optional label smoothing and EMA). The pipeline is config‑driven, easy to inspect, and robust across CUDA/MPS/CPU.

Approach Overview
- Encoder: Hugging Face `AutoModel` backbone returns token hidden states `[B, L, d]`.
- Head: Two interchangeable span‑aware predictors:
  - Conditional Pointer: Predicts start logits, extracts top‑K starts, and conditions end logits on those starts via attention.
  - Biaffine Span: Jointly scores (start i, offset j‑i) within a max‑answer band.
- Training: Cross‑entropy loss with ignore semantics for windows without gold spans; linear warmup scheduler; gradient clipping; optional EMA and label smoothing.
- Inference: Sliding windows with offsets; decode by combining start/end scores with a length penalty; aggregate across windows.

Repository Layout
- `src/data_processing.py`: Preprocess SQuAD JSON → HF `DatasetDict` with sliding windows and aligned spans.
- `src/augment_backtranslation.py`: Back-translation data augmentation using Helsinki-NLP MarianMT to create paraphrased questions/contexts.
- `src/inspect_processed.py`, `src/utils/inspect_dataset.py`: Inspect dataset shapes, masks, span coverage, and token previews.
- `src/data_wrappers.py`: PyTorch dataset wrapper that picks one gold span per window; emits `-100` when no span fits the window.
- `src/models/encoders.py`: `HFEncoder` thin wrapper around `AutoModel`/`AutoConfig` with optional `token_type_ids` passthrough.
- `src/models/heads.py`: `ConditionalPointerHead`, `BiaffineSpanHead` with numerically stable masking.
- `src/models/qa_model.py`: `QASpanProposer` assembly and training loss (pointer/biaffine with label smoothing and ignore handling).
- `src/utils/`: Schedulers, label smoothing, EMA, seeding, CSV logging.
- `src/train.py`: Trainer with device/AMP handling, EMA, CSV logs, and checkpoint writer (encoder, tokenizer, QA head, metadata).
- `src/predict.py`: Window builder + model inference + span decoding; reads metadata from checkpoint.
- `config/*.yaml`: Configs for encoder, windowing, training, and logging.
- `src/evaluate-v2.0.py`, `src/eval_wrap.py`: Official scorer + wrapper for EM/F1.

Data Pipeline (Detailed)
- Input: SQuAD v1.1 JSON (`data/train-v1.1.json`, `data/dev-v1.1.json`).
- Tokenization: Fast tokenizer with sliding windows; `truncation="only_second"` ensures context is truncated while full question is kept.
- Offsets and sequence IDs:
  - `return_offsets_mapping=True` and `return_overflowing_tokens=True` produce per‑window `offset_mapping` and `sequence_ids`.
  - Offsets for non‑context tokens (question/pad) are nulled, leaving only context `(start_char, end_char)` pairs.
- Alignment: For each gold answer `(answer_start, answer_end)` and each window, we map to token indices if the entire span lies within the window’s context region. If not, this window’s `answers` list remains empty.
- Output schema (per window):
  - `input_ids`, `attention_mask`, optional `token_type_ids` (depends on encoder family), `answers: List[{start_token_index, end_token_index, answer_text}]`.
- Saving: The dataset is saved via `DatasetDict.save_to_disk(processed_dir)`; loaded with `DatasetDict.load_from_disk(processed_dir)["train"]`.

Training Dataset Wrapper
- `SquadWindowDataset`:
  - On `__getitem__`, randomly selects one gold span from `answers` (if any). Otherwise sets `start_positions=end_positions=-100` (ignored in loss).
  - Defensive checks: Ensures selected indices are within length and lie on non‑padding tokens; otherwise marks as `-100`.
  - Returns tensors for `input_ids`, `attention_mask`, `token_type_ids` (zeros if absent), and labels.
- `qa_collate` stacks keys into batches with shape `[B, L]` for inputs and `[B]` for labels.

Model Architecture
- Encoder (`HFEncoder`):
  - Loads `AutoConfig` and `AutoModel` from `encoder_name`.
  - Passes `token_type_ids` only if the backbone supports it.
  - Returns `last_hidden_state` of shape `[B, L, d]`.
- Conditional Pointer Head:
  - Start logits: `start_logits = W_s H` masked over padding (`MASK_VAL = −1e2`).
  - Top‑K starts: `softmax(start_logits)` → pick `K` indices per batch item.
  - Conditioned end logits: For gathered start embeddings `S`:
    - Project to queries `Q= W_q S`; project tokens to keys/values `K_=W_k H`, `V=W_v H`.
    - Attention over tokens: `attn = softmax(Q K_^T / sqrt(d))`, context `C = attn V`.
    - Compare with tokens: `end_scores_k = (C W_cmp) · (H W_cmp)`;
    - Mixture over K using normalized top‑K start probabilities → `end_logits`.
  - Complexity: `O(B K L d)`; lowering `topk_start` reduces compute.
- Biaffine Span Head:
  - For each offset `off in [0, max_answer_len)`:
    - Bilinear term: `(H U) ⊙ roll(H, off)` plus linear boundary features.
    - Band mask: only tokens with both start and end inside attention mask are scored.
  - Returns scores `[B, L, max_answer_len]`.
- Masking Stability:
  - We use a moderate negative mask (`−1e2`) to avoid MPS/CPU underflow/overflow that can inflate cross‑entropy.

Losses and Stabilizers
- Pointer loss: `CE(start_logits, start_positions)` and `CE(end_logits, end_positions)`, averaged.
- Biaffine loss: For valid rows, `CE(span_scores[start_positions, :], offsets)` with `offset = end - start` clamped to band.
- Ignore semantics: `ignore_index = −100` supported across both heads; when no valid rows exist, the loss is a differentiable zero.
- Label smoothing: Implemented as `label_smoothing_ce` in `utils/smoothing.py`, supports ignore, mean/sum/none reductions.
- Gradient clipping: `clip_grad_norm_(..., max_grad_norm)` applied at each optimizer step.
- Scheduler: Linear warmup via `get_linear_schedule_with_warmup`.
  - Total steps computed as `ceil(steps_per_epoch * epochs / grad_accum_steps)` to align warmup and scheduler with optimizer steps.

Devices and Mixed Precision
- Device priority: CUDA → MPS → CPU.
- AMP: Enabled only on CUDA with `autocast` and `GradScaler`; disabled on MPS/CPU for stability.
- Dataloader: `pin_memory=True` only on CUDA; `num_workers=0` by default on macOS.

EMA (Optional)
- Maintains a moving average over trainable parameters.
- At save time, we temporarily copy EMA weights into the model to save the stabilized checkpoint, then restore original parameters.

Checkpoints and Metadata
- Layout inside a checkpoint folder (e.g., `outputs/.../final/`):
  - Encoder weights: `config.json`, `pytorch_model.bin`, etc. saved via `save_pretrained`.
  - Tokenizer files saved via `save_pretrained`.
  - QA head state dict: `qa_head.pt` (the full model state is saved; loaded with `strict=False`).
  - QA metadata: `qa_config.json` captures `head_type`, `max_answer_len`, and `topk_start` when using the pointer head.

Inference and Decoding
- Window building (`predict.py`): Same tokenization as preprocessing; offsets are set to `None` for non‑context tokens.
- Model forward: Produces `start_logits`/`end_logits` (pointer) or `span_scores` (biaffine).
- Candidate generation (`decode.py`):
  - Take top‑K start and top‑K end indices by logit value.
  - Filter invalid spans (`j < i`, length > `max_answer_len`, offsets `None`).
  - Score: `s[i] + e[j] − alpha * length` (length penalty α helps avoid overly long spans).
  - Convert token indices to char spans using window offsets, aggregate over all windows, and choose the best per question.

Configuration Reference
- `seed`: Random seed for Python/NumPy/PyTorch.
- `processed_dir`: Path to HF `DatasetDict` saved on disk.
- `encoder_name`: HF model name or local directory with a saved encoder.
- `max_length`: Token sequence length (smaller = faster, but more windows).
- `doc_stride`: Overlap between sliding windows (smaller stride = more windows).
- `max_answer_len`: Maximum answer token length used in training and decoding.
- `head_type`: `pointer` or `biaffine`.
- `topk_start`: Pointer head’s K for conditional end prediction (compute scales with K).
- `epochs`, `train_batch_size`, `grad_accum_steps`: Training schedule and batch size.
- `lr`, `weight_decay`, `warmup_ratio`, `max_grad_norm`: Optimization hyperparameters.
- `amp`: Mixed precision (CUDA only). Ignored on MPS/CPU.
- `label_smoothing`: 0.0–0.1 typical; turn on after basic stability is confirmed.
- `ema`, `ema_decay`: Exponential moving average of model parameters.
- `output_dir`: Where to save checkpoints and logs.
- `log_interval`, `save_every_steps`: Logging/saving cadence.
- `num_workers`: Dataloader workers (0 on macOS/MPS recommended).

Quick Start
- Install: `pip install -r requirements.txt`
- Preprocess: `python src/data_processing.py --input_file data/train-v1.1.json --output_dir data/processed_dataset`
- Train: `python src/train.py --config config/deberta_base_pointer.yaml`
- Predict: `python src/predict.py --checkpoint_dir outputs/deberta_v3_small_pointer_fast/final --dev_json data/dev-v1.1.json --output predictions.json`
- Evaluate: `python src/eval_wrap.py --dev_json data/dev-v1.1.json --pred_json predictions.json --evaluator_path src/evaluate-v2.0.py`

Data Augmentation via Back-Translation
- Purpose: Improve model robustness by exposing it to paraphrased questions and contexts through back-translation using Helsinki-NLP MarianMT models.
- Script: `src/augment_backtranslation.py` — translates English → pivot language (de/fr/es) → English to create linguistic variations.
- Features:
  - Automatic answer alignment: After back-translation, the script finds and verifies answer positions in the paraphrased context using character offset mapping.
  - Quality filtering: Examples where answers cannot be reliably found in back-translated contexts are automatically filtered out.
  - Configurable languages: Support for German (de), French (fr), and Spanish (es) as pivot languages.
  - Incremental augmentation: Augment a subset of examples per language to control dataset size.
- Usage:
  ```bash
  # Single language (German) with 100 examples
  python src/augment_backtranslation.py \
    --input data/train-v1.1.json \
    --output data/train-augmented.json \
    --languages de \
    --max-examples 100
  
  # Multiple languages with more examples
  python src/augment_backtranslation.py \
    --input data/train-v1.1.json \
    --output data/train-augmented-multi.json \
    --languages de fr es \
    --max-examples 500 \
    --device cuda  # or mps, cpu
  ```
- Workflow:
  1. Augment: Generate back-translated examples
  2. Preprocess: `python src/data_processing.py --input_file data/train-augmented.json --output_dir data/processed_augmented`
  3. Train: Use augmented data with your existing config
  4. Evaluate: Compare performance with/without augmentation
- Performance Impact (tested on SQuAD v1.1 subset):
  - Baseline (1,177 questions): EM=70.86%, F1=76.88%
  - With German augmentation (+78 questions, 6.6% increase): EM=72.19%, F1=77.72%
  - Improvement: +1.33% EM, +0.84% F1
  - Success rate: ~78% of back-translation attempts successfully preserve answer alignment
- Recommendations:
  - Start with a subset (100-500 examples per language) to test effectiveness
  - Use multiple pivot languages for maximum diversity
  - Expected improvement: 1-3% F1 with moderate augmentation
  - Augmentation is most effective when training data is limited

Performance Tips
- Mac (MPS): Keep `num_workers: 0`; AMP is disabled by design; prefer smaller `max_length` and `topk_start` for speed.
- CUDA: Enable AMP; increase batch size; consider `topk_start: 3–5`.
- Windowing: `max_length=192, doc_stride=64` is a good speed/accuracy trade‑off for quick runs.

Troubleshooting
- Huge initial loss (hundreds): Update to this codebase (stable masking), reprocess data to match `max_length`, ensure labels don’t touch padding.
- LR = 0 at start: Expected during warmup; it increases after a few steps.
- Mismatch head type at inference: Ensure `qa_config.json` exists; otherwise `predict.py` defaults to pointer with reasonable parameters.
- Missing `token_type_ids`: Expected for DeBERTa/Roberta; code handles it.

Changelog (Key Fixes)
- Stable masking value (−1e2) across heads to avoid MPS/CPU/AMP numeric issues.
- Defensive label handling in `data_wrappers.py` to ignore spans that touch padding or fall outside `[0, L)`.
- Pointer loss guard when all labels are `-100` (differentiable zero instead of NaN).
- Total step computation uses ceiling division with grad accumulation to align scheduler with optimizer steps.
- Preprocessing now honors `max_length`/`doc_stride` from YAML; includes `token_type_ids` only if produced by the tokenizer.
- `num_workers` robust parsing in training (non‑numeric → 0 on macOS/MPS/CPU).
- Checkpoint metadata (`qa_config.json`) is saved and used for consistent inference.

License
- The repository includes the official SQuAD v2.0 evaluation script for convenience. SQuAD data is subject to its own license.
