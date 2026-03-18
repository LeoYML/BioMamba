#!/usr/bin/env python3
"""
Standalone Continue Pre-Training on PubMed-MEDLINE.

Usage:
  python -m ft_biomamba.run_cpt
  python -m ft_biomamba.run_cpt --model mamba2-370m --epochs 5 --lr 3e-5
"""

import argparse
import random
import torch

from .config import CPTConfig, MAMBA2_MODELS, PROJECT_ROOT
from .model import load_model, load_tokenizer
from .data import prepare_cpt_data, prepare_cpt_data_packed
from .trainer import Trainer


def main():
    p = argparse.ArgumentParser(description="BioMamba CPT")
    p.add_argument("--model", default="mamba2-130m", choices=list(MAMBA2_MODELS.keys()))
    p.add_argument("--model_path", default=None, help="Resume from checkpoint")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--data_dir", default=os.path.join(PROJECT_ROOT, "data"))
    p.add_argument("--output_dir", default=os.path.join(PROJECT_ROOT, "checkpoints"))
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--packed", action="store_true", default=False,
                   help="Use packed sequences from large PubMed dataset")
    p.add_argument("--max_samples", type=int, default=2_000_000,
                   help="Max samples to load from large dataset (packed mode)")
    p.add_argument("--dataset", default="MedRAG/pubmed",
                   help="HuggingFace dataset for packed mode")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = load_tokenizer()
    cfg = CPTConfig(
        model_name=args.model, model_path=args.model_path,
        num_epochs=args.epochs, lr=args.lr, max_length=args.max_length,
        batch_size=args.batch_size, accumulation_steps=args.accum,
        data_dir=args.data_dir, output_dir=args.output_dir,
        gpu_id=args.gpu_id, seed=args.seed,
    )

    if args.packed:
        data = prepare_cpt_data_packed(
            tokenizer, cfg.data_dir, cfg.max_length,
            max_samples=args.max_samples,
            test_size=cfg.test_size, num_proc=cfg.num_proc,
            seed=cfg.seed, dataset_name=args.dataset,
        )
    else:
        data = prepare_cpt_data(tokenizer, cfg.data_dir, cfg.max_length,
                                cfg.test_size, cfg.num_proc, cfg.seed)
    model = load_model(cfg.resolve_model_path(),
                       device="cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(
        model, tokenizer,
        train_dataset=data["train"], eval_dataset=data["test"],
        mode="cpt", batch_size=cfg.resolve_batch_size(),
        accumulation_steps=cfg.accumulation_steps,
        lr=cfg.lr, weight_decay=cfg.weight_decay,
        num_epochs=cfg.num_epochs, warmup_ratio=cfg.warmup_ratio,
        stable_ratio=cfg.stable_ratio, decay_ratio=cfg.decay_ratio,
        min_lr_ratio=cfg.min_lr_ratio, max_grad_norm=cfg.max_grad_norm,
        scheduler_type="wsd", use_ema=cfg.use_ema, ema_decay=cfg.ema_decay,
        label_smoothing=cfg.label_smoothing, layer_lr_decay=cfg.layer_lr_decay,
        bf16=cfg.bf16, output_dir=cfg.output_dir,
        logging_steps=cfg.logging_steps, eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps, gpu_id=cfg.gpu_id,
        run_name=f"biomamba_cpt_{args.model}",
    )
    trainer.train()


if __name__ == "__main__":
    main()
