#!/usr/bin/env python3
"""
Run CPT training from single-document HF dataset (with attention_mask).

Supports both single-GPU and multi-GPU DDP training.

Single-GPU:
  python run_cpt_singledoc.py --data_dir ./data/pubmed_singledoc_mixed_5000k

Multi-GPU (e.g. 4x H100):
  torchrun --nproc_per_node=4 run_cpt_singledoc.py --data_dir ./data/pubmed_singledoc_mixed_5000k
"""
# --- Project root resolution (auto-generated) ---
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')
_sys.path.insert(0, _PROJECT_ROOT)
_os.chdir(_PROJECT_ROOT)
# --- End project root resolution ---

import argparse
import os
import random

import torch
import torch.distributed as dist

from datasets import load_from_disk

from ft_biomamba.config import CPTConfig
from ft_biomamba.model import load_model, load_tokenizer
from ft_biomamba.trainer import Trainer


def setup_distributed():
    """Initialize distributed training if launched via torchrun."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        return True
    return False


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description="BioMamba CPT (single-doc)")
    p.add_argument("--model", default="mamba2-130m")
    p.add_argument("--resume_from", default=None, help="Resume from checkpoint path")
    p.add_argument("--data_dir", default="./data/pubmed_singledoc_mixed_5000k")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--layer_lr_decay", type=float, default=0.9)
    p.add_argument("--output_dir", default="./checkpoints/singledoc")
    p.add_argument("--gpu_id", type=int, default=0, help="GPU id for single-GPU mode (ignored in DDP)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grad_ckpt", action="store_true", help="Enable gradient checkpointing")
    args = p.parse_args()

    is_distributed = setup_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = (local_rank == 0) or not is_distributed

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load data
    if is_main:
        print(f"Loading single-doc data from {args.data_dir}...")
    train_ds = load_from_disk(os.path.join(args.data_dir, "train"))
    test_ds = load_from_disk(os.path.join(args.data_dir, "test"))
    train_ds.set_format("torch")
    test_ds.set_format("torch")

    # Count actual tokens
    if is_main:
        sample_n = min(5000, len(train_ds))
        avg_len = sum(
            sum(train_ds[i]["attention_mask"]) if isinstance(train_ds[i]["attention_mask"], list)
            else int(train_ds[i]["attention_mask"].sum())
            for i in range(sample_n)
        ) / sample_n
        total_real_tokens = int(avg_len * len(train_ds))
        print(f"  Train: {len(train_ds):,} sequences, ~{total_real_tokens:,} real tokens (avg_len={avg_len:.0f})")
        print(f"  Test:  {len(test_ds):,} sequences")

    # Load model
    tokenizer = load_tokenizer()
    model_path = args.resume_from or f"state-spaces/{args.model}"
    if is_main:
        print(f"Loading model from {model_path}...")
    model = load_model(model_path, device="cpu",  # load to CPU first, Trainer moves to GPU
                       gradient_checkpointing=args.grad_ckpt)

    cfg = CPTConfig(
        model_name=args.model,
        num_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        accumulation_steps=args.accum,
        label_smoothing=args.label_smoothing,
        layer_lr_decay=args.layer_lr_decay,
        output_dir=args.output_dir,
        gpu_id=args.gpu_id,
        seed=args.seed,
    )

    run_name = f"biomamba_cpt_singledoc_{args.model}"
    trainer = Trainer(
        model, tokenizer,
        train_dataset=train_ds, eval_dataset=test_ds,
        mode="cpt", batch_size=args.batch_size,
        accumulation_steps=args.accum,
        lr=cfg.lr, weight_decay=cfg.weight_decay,
        num_epochs=cfg.num_epochs, warmup_ratio=cfg.warmup_ratio,
        stable_ratio=cfg.stable_ratio, decay_ratio=cfg.decay_ratio,
        min_lr_ratio=cfg.min_lr_ratio, max_grad_norm=cfg.max_grad_norm,
        scheduler_type="wsd", use_ema=cfg.use_ema, ema_decay=cfg.ema_decay,
        label_smoothing=cfg.label_smoothing, layer_lr_decay=cfg.layer_lr_decay,
        bf16=cfg.bf16, output_dir=cfg.output_dir,
        logging_steps=cfg.logging_steps, eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps, gpu_id=cfg.gpu_id,
        run_name=run_name,
    )
    trainer.train()

    cleanup_distributed()


if __name__ == "__main__":
    main()
