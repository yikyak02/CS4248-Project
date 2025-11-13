# CS4248 Question Answering Project - Code Structure

## Overview

This project implements extractive question answering on the SQuAD 1.1 dataset using DeBERTa-v3-large as the encoder with two custom QA head architectures:
- **Conditional Pointer Head**: Predicts start position, then end position conditioned on top-K starts
- **Biaffine Span Head**: Jointly scores all valid spans using biaffine attention

## Project Structure

```
CS4248-Project/
├── config/
│   └── deberta_base_pointer.yaml    # Training configuration
├── data/
│   ├── train-v1.1.json              # SQuAD 1.1 training data
│   ├── dev-v1.1.json                # SQuAD 1.1 dev data
│   ├── processed_dataset/           # Preprocessed training data
│   └── processed_dev/               # Preprocessed validation data
├── src/
│   ├── models/
│   │   ├── encoders.py              # DeBERTa encoder wrapper
│   │   ├── heads.py                 # QA head implementations
│   │   └── qa_model.py              # Main QA model (encoder + head)
│   ├── utils/
│   │   ├── ema.py                   # Exponential moving average
│   │   ├── logging.py               # CSV logger
│   │   ├── schedule.py              # Learning rate schedulers
│   │   ├── seed.py                  # Random seed utilities
│   │   └── smoothing.py             # Label smoothing
│   ├── augment_backtranslation.py   # Data augmentation via back-translation
│   ├── data_processing.py           # Preprocess SQuAD data
│   ├── data_wrappers.py             # PyTorch Dataset classes
│   ├── decode.py                    # Decoding logic for span extraction
│   ├── train.py                     # Training script
│   ├── predict.py                   # Inference script
│   ├── evaluate-v2.0.py             # Official SQuAD evaluation
│   ├── eval_wrap.py                 # Evaluation wrapper
│   └── inspect_processed.py         # Utility to inspect processed datasets
├── outputs/                         # Trained model checkpoints
└── predictions/                     # Prediction outputs
```

## Key Components

### 1. Data Augmentation (`augment_backtranslation.py`)

Generates paraphrased training examples via back-translation:
- Uses Helsinki-NLP MarianMT models for translation
- Supports multiple pivot languages: de, fr, es, ru, zh
- Batched translation for efficiency (12-24x speedup)
- Automatic answer alignment and quality filtering
- Preserves SQuAD JSON format for downstream processing

**Usage:**
```bash
python src/augment_backtranslation.py \
  --input data/train-v1.1.json \
  --output data/train-augmented-full.json \
  --languages de fr es ru zh \
  --max-examples 87599 \
  --batch-size 64 \
  --device cuda
```

### 2. Data Processing (`data_processing.py`)

Converts SQuAD JSON to tokenized windows with sliding window approach:
- Tokenizes question + context pairs
- Creates overlapping windows for long contexts
- Aligns answer spans to token positions
- Handles edge cases with robust error handling
- Saves preprocessed data in HuggingFace Datasets format

**Usage:**
```bash
python src/data_processing.py \
  --input_file data/train-v1.1.json \
  --output_dir data/processed_dataset
```

### 3. Model Architecture

#### Encoder (`encoders.py`)
- Wrapper around HuggingFace DeBERTa-v3-large
- Extracts contextualized token representations

#### QA Heads (`heads.py`)

**ConditionalPointerHead:**
- Predicts start logits independently
- Gets top-K start candidates
- Uses attention to condition end prediction on start
- Output: `(start_logits [B, L], end_logits [B, L])`

**BiaffineSpanHead:**
- Scores (start_position, offset) pairs jointly
- Uses biaffine attention: `score = start^T · U · end + proj([start; end])`
- More efficient than all L² (start, end) pairs
- Output: `span_scores [B, L, max_answer_len]`

#### Main Model (`qa_model.py`)
- Combines encoder + head
- Implements loss computation for both head types
- Handles label smoothing and ignore indices

### 4. Training (`train.py`)

Features:
- Mixed precision training (AMP)
- Gradient accumulation
- Learning rate warmup and decay
- Early stopping based on validation loss
- Exponential moving average (optional)
- Saves best checkpoint and training metrics

**Usage:**
```bash
python src/train.py config/deberta_base_pointer.yaml
```

### 5. Prediction (`predict.py`)

Features:
- Sliding window inference for long contexts
- Aggregates predictions across windows
- Supports both pointer and biaffine heads
- Outputs SQuAD-format predictions

**Usage:**
```bash
python src/predict.py \
  --checkpoint_dir outputs/deberta_large_pointer/best \
  --dev_json data/dev-v1.1.json \
  --output predictions/dev_predictions.json \
  --batch_size 16 \
  --max_length 384 \
  --doc_stride 128
```

### 6. Evaluation (`evaluate-v2.0.py`)

Official SQuAD evaluation script:
- Computes Exact Match (EM) and F1 scores
- Handles normalization and tokenization

**Usage:**
```bash
python src/evaluate-v2.0.py \
  data/dev-v1.1.json \
  predictions/dev_predictions.json
```

## Configuration

All hyperparameters are in `config/deberta_base_pointer.yaml`:

```yaml
# Model
encoder_name: microsoft/deberta-v3-large
head_type: pointer  # or 'biaffine'
topk_start: 10      # For pointer head
max_answer_len: 30  # For biaffine head

# Data
max_length: 384
doc_stride: 128

# Training
epochs: 10
train_batch_size: 16
grad_accum_steps: 4
lr: 3e-5
dropout: 0.1
label_smoothing: 0.0

# Early stopping
early_stopping: true
patience: 3
```

## Training Pipeline

1. **[Optional] Augment training data with back-translation:**
   ```bash
   python src/augment_backtranslation.py \
     --input data/train-v1.1.json \
     --output data/train-augmented-full.json \
     --languages de fr es ru zh \
     --max-examples 87599 \
     --batch-size 64 \
     --device cuda
   ```

2. **Preprocess data:**
   ```bash
   # Original data
   python src/data_processing.py \
     --input_file data/train-v1.1.json \
     --output_dir data/processed_dataset
   
   # Or augmented data
   python src/data_processing.py \
     --input_file data/train-augmented-full.json \
     --output_dir data/processed_train_augmented
   
   # Dev data
   python src/data_processing.py \
     --input_file data/dev-v1.1.json \
     --output_dir data/processed_dev
   ```

3. **Train model:**
   ```bash
   python src/train.py config/deberta_base_pointer.yaml
   ```

4. **Generate predictions:**
   ```bash
   python src/predict.py \
     --checkpoint_dir outputs/deberta_large_pointer/best \
     --dev_json data/dev-v1.1.json \
     --output predictions/dev_predictions.json
   ```

5. **Evaluate:**
   ```bash
   python src/evaluate-v2.0.py \
     data/dev-v1.1.json \
     predictions/dev_predictions.json
   ```

## Model Variants

### Pointer Head
- **Advantages**: Simpler, faster inference
- **Architecture**: Independent start/end with conditioning
- **Expected Performance**: 88-90% EM, 92-95% F1

### Biaffine Head
- **Advantages**: Joint span modeling, better boundary detection
- **Architecture**: Biaffine attention over (start, offset) pairs
- **Expected Performance**: 78-82% EM, 85-88% F1

## Key Implementation Details

### Sliding Windows
- Long contexts split into overlapping windows
- Window size: 384 tokens (question + partial context)
- Stride: 128 tokens (overlap of 256 tokens)
- Predictions aggregated across windows by max score

### Loss Computation

**Pointer Head:**
```python
loss = 0.5 * (cross_entropy(start_logits, start_pos) + 
              cross_entropy(end_logits, end_pos))
```

**Biaffine Head:**
```python
offset = end_pos - start_pos
scores_at_start = span_scores[batch_idx, start_pos, :]
loss = cross_entropy(scores_at_start, offset)
```

### Decoding

**Pointer Head:**
- Get top-K start positions
- Get top-K end positions  
- Try all valid (start, end) pairs
- Filter: end >= start, length <= max_answer_len
- Return highest scoring span

**Biaffine Head:**
- Flatten [L, M] scores
- Get top-K (start, offset) pairs
- Filter out question/padding tokens
- Convert to character spans
- Return highest scoring span

## Performance Tips

1. **Use epoch 1 checkpoint** if overfitting (val loss increases after epoch 1)
2. **Lower learning rate** (1e-5) for more stable training
3. **Increase dropout** (0.2) for better generalization
4. **Use doc_stride=128** for good coverage/speed trade-off
5. **Remove length penalty** during inference (set to 0.0)

## Common Issues

### Overfitting
**Symptom:** Train loss decreases but val loss increases  
**Solution:** Use lower LR (1e-5), higher dropout (0.2), early stopping

### Low F1 Score
**Symptom:** Predictions are plausible but wrong  
**Solution:** Check using best epoch (lowest val loss), not last epoch

### Out of Memory
**Symptom:** CUDA OOM during training  
**Solution:** Reduce batch size, increase grad_accum_steps to maintain effective batch size

## Dependencies

```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.14.0
tqdm>=4.65.0
PyYAML>=6.0
```

## Citation

This project builds on:
- DeBERTa: He et al., "DeBERTa: Decoding-enhanced BERT with Disentangled Attention", ICLR 2021
- SQuAD: Rajpurkar et al., "SQuAD: 100,000+ Questions for Machine Comprehension of Text", EMNLP 2016
