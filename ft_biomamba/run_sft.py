#!/usr/bin/env python3
"""
Standalone SFT on PubMedQA (+ optional BioASQ / MedQA).

Usage:
  python -m ft_biomamba.run_sft --model_path ./checkpoints/biomamba_cpt_mamba2-130m/best_model
  python -m ft_biomamba.run_sft --model_path state-spaces/mamba2-130m --use_lora
  python -m ft_biomamba.run_sft --model_path ./checkpoints/biomamba_cpt_mamba2-130m/best_model --mix_medqa
"""

import argparse
import random
import torch

from .config import SFTConfig, MAMBA2_MODELS, PROJECT_ROOT
from .model import load_model, load_tokenizer, inject_lora, count_parameters
from .data import prepare_sft_data
from .trainer import Trainer


def main():
    p = argparse.ArgumentParser(description="BioMamba SFT")
    p.add_argument("--model", default="mamba2-130m", choices=list(MAMBA2_MODELS.keys()))
    p.add_argument("--model_path", required=True, help="Path to CPT checkpoint or HF model")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--mix_bioasq", action="store_true", default=True)
    p.add_argument("--no_mix_bioasq", dest="mix_bioasq", action="store_false")
    p.add_argument("--bioasq_data_path", default=None)
    p.add_argument("--bioasq_train_ratio", type=float, default=0.3)
    p.add_argument("--bioasq_only", action="store_true", default=False,
                   help="Train on BioASQ data only (no PubMedQA)")
    p.add_argument("--mix_medqa", action="store_true", default=False)
    p.add_argument("--no_mix_medqa", dest="mix_medqa", action="store_false")
    p.add_argument("--medqa_train_ratio", type=float, default=0.2)
    p.add_argument("--data_dir", default=os.path.join(PROJECT_ROOT, "data"))
    p.add_argument("--output_dir", default=os.path.join(PROJECT_ROOT, "checkpoints"))
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = load_tokenizer()
    data = prepare_sft_data(
        tokenizer, data_dir=args.data_dir, max_length=args.max_length,
        mix_bioasq=args.mix_bioasq, bioasq_data_path=args.bioasq_data_path,
        bioasq_train_ratio=args.bioasq_train_ratio,
        bioasq_only=args.bioasq_only,
        mix_medqa=args.mix_medqa, medqa_train_ratio=args.medqa_train_ratio,
        seed=args.seed,
    )

    model = load_model(args.model_path, device="cuda" if torch.cuda.is_available() else "cpu")

    trainable = None
    lora_tag = "_full"
    if args.use_lora:
        lora_params = inject_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
        trainable = lora_params
        lora_tag = f"_lora_r{args.lora_rank}"

    trainer = Trainer(
        model, tokenizer,
        train_dataset=data["train"], eval_dataset=data["validation"],
        mode="sft", batch_size=args.batch_size,
        accumulation_steps=args.accum, lr=args.lr,
        weight_decay=0.01, num_epochs=args.epochs,
        warmup_ratio=0.1, max_grad_norm=1.0,
        scheduler_type="cosine", bf16=True,
        output_dir=args.output_dir,
        logging_steps=10, eval_steps=50, save_steps=200,
        gpu_id=args.gpu_id,
        trainable_params=trainable,
        run_name=f"biomamba_sft_{args.model}{lora_tag}",
        patience=5,
    )
    trainer.train()


if __name__ == "__main__":
    main()
