#!/usr/bin/env python3
"""
End-to-end BioMamba fine-tuning pipeline:
  Stage 1 - Continue Pre-Training on PubMed-MEDLINE
  Stage 2 - Supervised Fine-Tuning on PubMedQA (+BioASQ)
  Stage 3 - Evaluation on BioASQ (with base-model comparison)

Usage:
  python -m ft_biomamba.run_pipeline                         # full pipeline
  python -m ft_biomamba.run_pipeline --skip_cpt              # SFT + eval only
  python -m ft_biomamba.run_pipeline --skip_cpt --skip_sft   # eval only
  python -m ft_biomamba.run_pipeline --model mamba2-370m     # larger model
"""

import argparse
import random
import os

import torch

from .config import CPTConfig, SFTConfig, EvalConfig, MAMBA2_MODELS, PROJECT_ROOT
from .model import load_model, load_tokenizer, inject_lora, count_parameters
from .data import prepare_cpt_data, prepare_sft_data, load_bioasq_eval
from .trainer import Trainer
from .evaluate import run_evaluation


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="BioMamba fine-tuning pipeline")

    # Model
    p.add_argument("--model", default="mamba2-130m", choices=list(MAMBA2_MODELS.keys()))
    p.add_argument("--cpt_checkpoint", default=None, help="Skip CPT, use this checkpoint for SFT")
    p.add_argument("--sft_checkpoint", default=None, help="Skip CPT+SFT, use this checkpoint for eval")

    # Stage control
    p.add_argument("--skip_cpt", action="store_true")
    p.add_argument("--skip_sft", action="store_true")
    p.add_argument("--skip_eval", action="store_true")

    # CPT overrides
    p.add_argument("--cpt_epochs", type=int, default=3)
    p.add_argument("--cpt_lr", type=float, default=5e-6)
    p.add_argument("--cpt_max_length", type=int, default=1024)
    p.add_argument("--cpt_batch_size", type=int, default=None)
    p.add_argument("--cpt_accum", type=int, default=8)

    # SFT overrides
    p.add_argument("--sft_epochs", type=int, default=5)
    p.add_argument("--sft_lr", type=float, default=2e-5)
    p.add_argument("--sft_max_length", type=int, default=1024)
    p.add_argument("--sft_batch_size", type=int, default=8)
    p.add_argument("--sft_accum", type=int, default=4)
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--mix_bioasq", action="store_true", default=True)
    p.add_argument("--no_mix_bioasq", dest="mix_bioasq", action="store_false")
    p.add_argument("--bioasq_data_path", default=None)
    p.add_argument("--bioasq_train_ratio", type=float, default=0.3)

    # Eval overrides
    p.add_argument("--eval_dataset", default="bioasq", choices=["bioasq", "pubmedqa"])
    p.add_argument("--eval_data_path", default=None)
    p.add_argument("--eval_max_samples", type=int, default=None)
    p.add_argument("--compare_base", action="store_true", default=True,
                   help="Compare fine-tuned model with base model")
    p.add_argument("--no_compare_base", dest="compare_base", action="store_false")

    # Common
    p.add_argument("--data_dir", default=os.path.join(PROJECT_ROOT, "data"))
    p.add_argument("--output_dir", default=os.path.join(PROJECT_ROOT, "checkpoints"))
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no_bf16", dest="bf16", action="store_false")

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    tokenizer = load_tokenizer()

    cpt_best = args.cpt_checkpoint
    sft_best = args.sft_checkpoint

    # ==================================================================
    # Stage 1: Continue Pre-Training
    # ==================================================================
    if not args.skip_cpt and sft_best is None and cpt_best is None:
        print("\n" + "=" * 65)
        print("  STAGE 1: Continue Pre-Training on PubMed-MEDLINE")
        print("=" * 65)

        cpt_cfg = CPTConfig(
            model_name=args.model,
            num_epochs=args.cpt_epochs,
            lr=args.cpt_lr,
            max_length=args.cpt_max_length,
            batch_size=args.cpt_batch_size,
            accumulation_steps=args.cpt_accum,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            gpu_id=args.gpu_id,
            seed=args.seed,
            bf16=args.bf16,
        )

        # Load data
        cpt_data = prepare_cpt_data(
            tokenizer, data_dir=cpt_cfg.data_dir,
            max_length=cpt_cfg.max_length, test_size=cpt_cfg.test_size,
            num_proc=cpt_cfg.num_proc, seed=cpt_cfg.seed,
        )

        # Load model
        model = load_model(cpt_cfg.resolve_model_path(), device="cuda" if torch.cuda.is_available() else "cpu")

        # Train
        trainer = Trainer(
            model, tokenizer,
            train_dataset=cpt_data["train"],
            eval_dataset=cpt_data["test"],
            mode="cpt",
            batch_size=cpt_cfg.resolve_batch_size(),
            accumulation_steps=cpt_cfg.accumulation_steps,
            lr=cpt_cfg.lr,
            weight_decay=cpt_cfg.weight_decay,
            num_epochs=cpt_cfg.num_epochs,
            warmup_ratio=cpt_cfg.warmup_ratio,
            stable_ratio=cpt_cfg.stable_ratio,
            decay_ratio=cpt_cfg.decay_ratio,
            min_lr_ratio=cpt_cfg.min_lr_ratio,
            max_grad_norm=cpt_cfg.max_grad_norm,
            scheduler_type="wsd",
            use_ema=cpt_cfg.use_ema,
            ema_decay=cpt_cfg.ema_decay,
            label_smoothing=cpt_cfg.label_smoothing,
            layer_lr_decay=cpt_cfg.layer_lr_decay,
            bf16=cpt_cfg.bf16,
            output_dir=cpt_cfg.output_dir,
            log_dir=cpt_cfg.log_dir,
            logging_steps=cpt_cfg.logging_steps,
            eval_steps=cpt_cfg.eval_steps,
            save_steps=cpt_cfg.save_steps,
            gpu_id=cpt_cfg.gpu_id,
            run_name=f"biomamba_cpt_{args.model}",
        )
        cpt_best = trainer.train()

        # Free memory
        del model, trainer
        torch.cuda.empty_cache()

    elif cpt_best is None and sft_best is None:
        # skip_cpt but no checkpoint: use base model for SFT
        cpt_best = MAMBA2_MODELS[args.model]["path"]
        print(f"[pipeline] Skipping CPT, using base model: {cpt_best}")

    # ==================================================================
    # Stage 2: Supervised Fine-Tuning
    # ==================================================================
    if not args.skip_sft and sft_best is None:
        print("\n" + "=" * 65)
        print("  STAGE 2: Supervised Fine-Tuning on PubMedQA")
        print("=" * 65)

        model_path = cpt_best
        print(f"  Loading from: {model_path}")

        # SFT data
        sft_data = prepare_sft_data(
            tokenizer, data_dir=args.data_dir,
            max_length=args.sft_max_length, num_proc=4,
            mix_bioasq=args.mix_bioasq,
            bioasq_data_path=args.bioasq_data_path,
            bioasq_train_ratio=args.bioasq_train_ratio,
            seed=args.seed,
        )

        # Load model
        model = load_model(model_path, device="cuda" if torch.cuda.is_available() else "cpu")

        # Optional LoRA
        trainable = None
        if args.use_lora:
            lora_params = inject_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
            trainable = lora_params
            total, trn = count_parameters(model)
            trn_lora = sum(p.numel() for p in lora_params)
            print(f"  LoRA injected: {trn_lora:,} params ({100*trn_lora/total:.2f}% of total)")

        # Train
        lora_tag = f"_lora_r{args.lora_rank}" if args.use_lora else "_full"
        trainer = Trainer(
            model, tokenizer,
            train_dataset=sft_data["train"],
            eval_dataset=sft_data["validation"],
            mode="sft",
            batch_size=args.sft_batch_size,
            accumulation_steps=args.sft_accum,
            lr=args.sft_lr,
            weight_decay=0.01,
            num_epochs=args.sft_epochs,
            warmup_ratio=0.1,
            max_grad_norm=1.0,
            scheduler_type="cosine",
            use_ema=False,
            label_smoothing=0.0,
            bf16=args.bf16,
            output_dir=args.output_dir,
            logging_steps=10,
            eval_steps=50,
            save_steps=200,
            gpu_id=args.gpu_id,
            trainable_params=trainable,
            run_name=f"biomamba_sft_{args.model}{lora_tag}",
            patience=5,
        )
        sft_best = trainer.train()

        # Free memory
        del model, trainer
        torch.cuda.empty_cache()

    # ==================================================================
    # Stage 3: Evaluation
    # ==================================================================
    if not args.skip_eval:
        print("\n" + "=" * 65)
        print("  STAGE 3: Evaluation on BioASQ")
        print("=" * 65)

        eval_cfg = EvalConfig(
            model_path=sft_best or cpt_best or MAMBA2_MODELS[args.model]["path"],
            base_model_path=MAMBA2_MODELS[args.model]["path"] if args.compare_base else None,
            dataset_name=args.eval_dataset,
            data_path=args.eval_data_path or os.path.join(args.data_dir, "bioasq_test"),
            max_length=args.sft_max_length,
            max_samples=args.eval_max_samples,
            gpu_id=args.gpu_id,
        )

        # Load eval dataset
        if args.eval_dataset == "bioasq":
            from .data import load_bioasq_eval
            eval_ds = load_bioasq_eval(eval_cfg.data_path, eval_cfg.split)
        else:
            from .data import load_pubmedqa_eval
            eval_ds = load_pubmedqa_eval(eval_cfg.split, args.seed)

        print(f"  Eval samples: {len(eval_ds)}")
        run_evaluation(eval_cfg, eval_ds)

    print("\n" + "=" * 65)
    print("  Pipeline complete!")
    print("=" * 65)


if __name__ == "__main__":
    main()
