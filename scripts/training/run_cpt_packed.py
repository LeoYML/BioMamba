#!/usr/bin/env python3
"""
Run CPT training from pre-packed numpy data.

Usage:
  python run_cpt_packed.py                     # train from base model
  python run_cpt_packed.py --resume_from ./checkpoints/packed/biomamba_cpt_packed_mamba2-130m/best_model
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

import numpy as np
import torch
from torch.utils.data import Dataset

from ft_biomamba.config import CPTConfig
from ft_biomamba.model import load_model, load_tokenizer
from ft_biomamba.trainer import Trainer


class NumpyPackedDataset(Dataset):
    """Wraps numpy mmap arrays as a torch Dataset."""
    def __init__(self, path):
        self.data = np.load(path, mmap_mode='r')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = torch.tensor(self.data[idx], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def main():
    p = argparse.ArgumentParser(description="BioMamba CPT (packed data)")
    p.add_argument("--model", default="mamba2-130m")
    p.add_argument("--resume_from", default=None, help="Resume from checkpoint path")
    p.add_argument("--data_dir", default="./data/pubmed_cpt_packed_1024_5000k")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--layer_lr_decay", type=float, default=0.95)
    p.add_argument("--output_dir", default="./checkpoints/packed")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load data
    print(f"Loading packed data from {args.data_dir}...")
    train_ds = NumpyPackedDataset(os.path.join(args.data_dir, "train_ids.npy"))
    test_ds = NumpyPackedDataset(os.path.join(args.data_dir, "test_ids.npy"))
    print(f"  Train: {len(train_ds):,} sequences ({len(train_ds) * 1024:,} tokens)")
    print(f"  Test:  {len(test_ds):,} sequences ({len(test_ds) * 1024:,} tokens)")

    # Load model
    tokenizer = load_tokenizer()
    model_path = args.resume_from or f"state-spaces/{args.model}"
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, device="cuda" if torch.cuda.is_available() else "cpu")

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

    run_name = f"biomamba_cpt_packed_{args.model}_lr{args.lr:.0e}"
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


if __name__ == "__main__":
    main()
