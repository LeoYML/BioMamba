#!/usr/bin/env python3
"""
Clinical SFT on MIMIC discharge notes.

Trains the CPT model to:
1. Complete clinical notes (given prefix → generate continuation)
2. Generate discharge summaries (given admission info → discharge sections)

Usage:
  # Full SFT (both tasks) on CPT model
  python -m ft_biomamba.run_mimic_sft \
    --model_path ./checkpoints/mixed_wiki/biomamba_cpt_singledoc_mamba2-130m/best_model

  # Discharge-only SFT
  python -m ft_biomamba.run_mimic_sft --model_path ... --task discharge

  # LoRA SFT (faster, less memory)
  python -m ft_biomamba.run_mimic_sft --model_path ... --use_lora --lora_rank 16 --lora_alpha 16
"""

import argparse
import random

import torch

from .config import MAMBA2_MODELS, PROJECT_ROOT
from .model import load_model, load_tokenizer, inject_lora
from .mimic_sft_data import prepare_mimic_sft_data
from .trainer import Trainer


def main():
    p = argparse.ArgumentParser(description="BioMamba Clinical SFT on MIMIC")

    # Model
    p.add_argument("--model", default="mamba2-130m", choices=list(MAMBA2_MODELS.keys()))
    p.add_argument("--model_path", required=True, help="Path to CPT checkpoint or HF model")

    # LoRA
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)  # alpha=rank → scaling=1.0

    # MIMIC data
    p.add_argument("--mimic_dir", default=os.path.join(PROJECT_ROOT, "data/mimic-iv-note/2.2/note"))
    p.add_argument("--mimic_version", default="iv", choices=["iii", "iv"])
    p.add_argument("--task", default="both", choices=["completion", "discharge", "both"],
                    help="SFT task type")
    p.add_argument("--max_train_samples", type=int, default=5000,
                    help="Max samples per task type")
    p.add_argument("--max_context_chars", type=int, default=3000)
    p.add_argument("--max_response_chars", type=int, default=2000)

    # Training
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--warmup_ratio", type=float, default=0.1)

    # Output
    p.add_argument("--data_dir", default=os.path.join(PROJECT_ROOT, "data"))
    p.add_argument("--output_dir", default=os.path.join(PROJECT_ROOT, "checkpoints/mimic_sft"))
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    # Seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Tokenizer & data
    tokenizer = load_tokenizer()
    data = prepare_mimic_sft_data(
        tokenizer,
        mimic_dir=args.mimic_dir,
        mimic_version=args.mimic_version,
        data_dir=args.data_dir,
        max_length=args.max_length,
        max_train_samples=args.max_train_samples,
        task=args.task,
        max_context_chars=args.max_context_chars,
        max_response_chars=args.max_response_chars,
        seed=args.seed,
    )

    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model_path, device=device)

    trainable = None
    lora_tag = "_full"
    if args.use_lora:
        lora_params = inject_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
        trainable = lora_params
        lora_tag = f"_lora_r{args.lora_rank}"

    # Train
    run_name = f"biomamba_mimic_sft_{args.model}_{args.task}{lora_tag}"
    trainer = Trainer(
        model, tokenizer,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        mode="sft",
        batch_size=args.batch_size,
        accumulation_steps=args.accum,
        lr=args.lr,
        weight_decay=0.01,
        num_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=1.0,
        scheduler_type="cosine",
        bf16=True,
        output_dir=args.output_dir,
        logging_steps=10,
        eval_steps=50,
        save_steps=200,
        gpu_id=args.gpu_id,
        trainable_params=trainable,
        run_name=run_name,
        patience=5,
    )
    best_path = trainer.train()
    print(f"\nBest model saved to: {best_path}")
    print(f"\nTo evaluate:")
    print(f"  python -m ft_biomamba.run_mimic_eval --model_path {best_path} --task completion")
    print(f"  python -m ft_biomamba.run_mimic_eval --model_path {best_path} --task discharge")


if __name__ == "__main__":
    main()
