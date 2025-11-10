# src/train.py
import os
import json
import sys
import math
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import DatasetDict
from contextlib import nullcontext

# --- Make repo/src importable so "data_wrappers", "models", "utils" resolve ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
# ------------------------------------------------------------------------------

from data_wrappers import SquadWindowDataset, qa_collate
from models.qa_model import QASpanProposer
from utils.seed import set_seed
from utils.schedule import build_linear_warmup, build_cosine_warmup
from utils.logging import CSVLogger, ensure_dir
from utils.ema import EMA


def load_processed_split(path: str, split: str = "train"):
    ds = DatasetDict.load_from_disk(path)
    if split not in ds:
        raise ValueError(f"Expected a '{split}' split in {path}")
    return ds[split]


@torch.no_grad()
def validate(model, loader, device, amp, amp_device):
    """Compute average validation loss"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        ctx = torch.amp.autocast(device_type=amp_device, enabled=amp) if amp else nullcontext()
        with ctx:
            out = model(**batch)
            total_loss += out["loss"].item()
        num_batches += 1
    
    model.train()
    return total_loss / max(1, num_batches)


def compute_total_steps(num_examples: int, batch_size: int, epochs: int, grad_accum: int) -> int:
    steps_per_epoch = math.ceil(num_examples / max(1, batch_size))
    total_forward_steps = steps_per_epoch * max(1, epochs)
    # Optimizer steps occur every grad_accum updates → ceil division to avoid undercount
    total_optimizer_steps = math.ceil(total_forward_steps / max(1, grad_accum))
    return max(1, total_optimizer_steps)


def save_checkpoint(dir_path: str, model, tokenizer):
    os.makedirs(dir_path, exist_ok=True)
    to_save = model.module if hasattr(model, "module") else model
    # Save encoder weights/tokenizer in HF format (loadable by AutoModel/AutoTokenizer)
    to_save.encoder.backbone.save_pretrained(dir_path)
    tokenizer.save_pretrained(dir_path)
    # Save the rest (QA head etc.) as a state_dict
    torch.save(to_save.state_dict(), os.path.join(dir_path, "qa_head.pt"))
    # Write lightweight QA config to aid inference
    qa_cfg = {
        "head_type": getattr(to_save, "head_type", "pointer"),
        "max_answer_len": int(getattr(to_save, "max_answer_len", 30)),
        "label_smoothing": float(getattr(to_save, "label_smoothing", 0.0)),
    }
    # add topk_start if pointer head
    try:
        if qa_cfg["head_type"] == "pointer" and hasattr(to_save, "head") and hasattr(to_save.head, "topk"):
            qa_cfg["topk_start"] = int(getattr(to_save.head, "topk", 5))
    except Exception:
        pass
    try:
        with open(os.path.join(dir_path, "qa_config.json"), "w") as f:
            json.dump(qa_cfg, f)
    except Exception:
        pass
    print(f"[checkpoint] saved -> {dir_path}")


def main(cfg_path: str):
    # ------------------ config ------------------
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # robust type-casting (tolerates quoted yaml values)
    seed            = int(cfg.get("seed", 42))
    processed_dir   = cfg["processed_dir"]
    val_processed_dir = cfg.get("val_processed_dir")  # optional validation data
    encoder_name    = cfg["encoder_name"]
    head_type       = cfg.get("head_type", "pointer")
    topk_start      = int(cfg.get("topk_start", 5))
    max_answer_len  = int(cfg.get("max_answer_len", 30))

    epochs          = int(cfg.get("epochs", 3))
    train_bs        = int(cfg.get("train_batch_size", 12))
    grad_accum      = int(cfg.get("grad_accum_steps", 1))
    lr              = float(cfg.get("lr", 3e-5))
    weight_decay    = float(cfg.get("weight_decay", 0.01))
    warmup_ratio    = float(cfg.get("warmup_ratio", 0.1))
    max_grad_norm   = float(cfg.get("max_grad_norm", 1.0))
    amp_cfg         = bool(cfg.get("amp", True))
    label_smooth    = float(cfg.get("label_smoothing", 0.0))
    dropout         = float(cfg.get("dropout", 0.1))
    use_ema         = bool(cfg.get("ema", False))
    ema_decay       = float(cfg.get("ema_decay", 0.999))
    scheduler_type  = cfg.get("scheduler", "linear")  # "linear" or "cosine"
    
    # Early stopping parameters
    use_early_stop  = bool(cfg.get("early_stopping", False))
    patience        = int(cfg.get("patience", 3))
    val_interval    = int(cfg.get("val_interval", 1))  # validate every N epochs
    
    # robust int parsing for num_workers
    _nw = cfg.get("num_workers", 0)
    try:
        num_workers = int(_nw)
    except Exception:
        num_workers = 0  # safest default on macOS/MPS/CPU

    out_dir         = cfg["output_dir"]
    log_interval    = int(cfg.get("log_interval", 100))
    save_every      = int(cfg.get("save_every_steps", 2000)) or 0

    set_seed(seed)

    # ---------------- device & AMP ----------------
    # Priority: CUDA -> MPS -> CPU; AMP only on CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
        amp = bool(amp_cfg)
        amp_device = "cuda"
        try:
            torch.backends.cuda.matmul.fp32_precision = 'high'
            torch.backends.cudnn.conv.fp32_precision = 'high'
        except Exception:
            pass
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        amp = False
        amp_device = "cpu"
    else:
        device = torch.device("cpu")
        amp = False
        amp_device = "cpu"

    print(f"[env] device={device}, amp={amp}, amp_device={amp_device}")

    # ------------------ data --------------------
    train_hf = load_processed_split(processed_dir, split="train")
    from transformers import AutoTokenizer  # import here to speed up module import
    tokenizer = AutoTokenizer.from_pretrained(encoder_name, use_fast=True)

    train_ds = SquadWindowDataset(train_hf, cls_token_id=tokenizer.cls_token_id or 0)
    train_loader = DataLoader(
        train_ds,
        batch_size=train_bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=qa_collate,
    )

    # Load validation data if provided
    val_loader = None
    if val_processed_dir:
        print(f"[data] loading validation from {val_processed_dir}")
        val_hf = load_processed_split(val_processed_dir, split="train")  # val data stored as "train" split
        val_ds = SquadWindowDataset(val_hf, cls_token_id=tokenizer.cls_token_id or 0)
        val_loader = DataLoader(
            val_ds,
            batch_size=train_bs * 2,  # larger batch for inference
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=qa_collate,
        )
        print(f"[data] validation set: {len(val_ds)} windows")

    # ------------------ model -------------------
    model = QASpanProposer(
        encoder_name=encoder_name,
        head_type=head_type,
        topk_start=topk_start,
        max_answer_len=max_answer_len,
        label_smoothing=label_smooth,
        dropout=dropout,
    ).to(device)

    # ---------------- optim/sched ---------------
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = compute_total_steps(
        num_examples=len(train_ds),
        batch_size=train_bs,
        epochs=epochs,
        grad_accum=grad_accum,
    )
    warmup_steps = int(warmup_ratio * total_steps)
    
    # Choose scheduler based on config
    if scheduler_type == "cosine":
        sched = build_cosine_warmup(opt, warmup_steps, total_steps)
    else:
        sched = build_linear_warmup(opt, warmup_steps, total_steps)

    # GradScaler only for CUDA AMP
    scaler = torch.amp.GradScaler('cuda') if (amp and device.type == "cuda") else None

    # ------------------ EMA (opt) ---------------
    ema = EMA(model, decay=ema_decay) if use_ema else None

    # ---------------- logging/io ----------------
    ensure_dir(out_dir)
    logger = CSVLogger(os.path.join(out_dir, "metrics_train.csv"))

    # ---------------- early stopping ------------
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_checkpoint_dir = os.path.join(out_dir, "best")

    # ---------------- train loop ----------------
    model.train()
    step = 0
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            ctx = torch.amp.autocast(device_type=amp_device, enabled=amp) if amp else nullcontext()
            with ctx:
                out = model(**batch)
                loss = out["loss"] / max(1, grad_accum)

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % max(1, grad_accum) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                if scaler is not None:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
                if ema is not None:
                    ema.update(model)

            epoch_loss += loss.item() * max(1, grad_accum)
            num_batches += 1

            if save_every and step > 0 and step % save_every == 0:
                if ema is not None:
                    to_save = model.module if hasattr(model, "module") else model
                    ema.store(to_save); ema.copy_to(to_save)
                    save_checkpoint(os.path.join(out_dir, f"step{step}"), model, tokenizer)
                    ema.restore(to_save)
                else:
                    save_checkpoint(os.path.join(out_dir, f"step{step}"), model, tokenizer)

            step += 1

        # Log epoch metrics
        avg_epoch_loss = epoch_loss / max(1, num_batches)
        lr_now = sched.get_last_lr()[0]
        logger.log({"epoch": epoch, "train_loss": avg_epoch_loss, "lr": lr_now})
        print(f"[train] epoch {epoch} | loss {avg_epoch_loss:.4f} | lr {lr_now:.6f}")

        # per-epoch validation (if enabled)
        if val_loader is not None and epoch % val_interval == 0:
            val_loss = validate(model, val_loader, device, amp, amp_device)
            logger.log({"epoch": epoch, "val_loss": val_loss})
            print(f"[val] epoch {epoch} | val_loss {val_loss:.4f}")
            
            # Early stopping logic
            if use_early_stop:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    # Save best checkpoint
                    print(f"[early_stop] New best validation loss: {val_loss:.4f}")
                    if ema is not None:
                        to_save = model.module if hasattr(model, "module") else model
                        ema.store(to_save); ema.copy_to(to_save)
                        save_checkpoint(best_checkpoint_dir, model, tokenizer)
                        ema.restore(to_save)
                    else:
                        save_checkpoint(best_checkpoint_dir, model, tokenizer)
                else:
                    epochs_no_improve += 1
                    print(f"[early_stop] No improvement for {epochs_no_improve}/{patience} epochs")
                    
                    if epochs_no_improve >= patience:
                        print(f"[early_stop] Stopping training at epoch {epoch}")
                        print(f"[early_stop] Best validation loss: {best_val_loss:.4f}")
                        return  # Exit training early

        # per-epoch save
        if ema is not None:
            to_save = model.module if hasattr(model, "module") else model
            ema.store(to_save); ema.copy_to(to_save)
            save_checkpoint(os.path.join(out_dir, f"ep{epoch}"), model, tokenizer)
            ema.restore(to_save)
        else:
            save_checkpoint(os.path.join(out_dir, f"ep{epoch}"), model, tokenizer)

    # final save
    final_dir = os.path.join(out_dir, "final")
    if ema is not None:
        to_save = model.module if hasattr(model, "module") else model
        ema.store(to_save); ema.copy_to(to_save)
        save_checkpoint(final_dir, model, tokenizer)
        ema.restore(to_save)
    else:
        save_checkpoint(final_dir, model, tokenizer)

    print("Training complete.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default="config/deberta_base_pointer.yaml",
        help="Path to YAML config (default: config/deberta_base_pointer.yaml)",
    )
    args = ap.parse_args()
    main(args.config)
