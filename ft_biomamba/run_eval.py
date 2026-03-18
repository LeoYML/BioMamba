#!/usr/bin/env python3
"""
Standalone evaluation on BioASQ / PubMedQA.

Usage:
  python -m ft_biomamba.run_eval --model_path ./checkpoints/biomamba_sft_.../best_model
  python -m ft_biomamba.run_eval --model_path ... --base_model state-spaces/mamba2-130m
"""

import argparse
import os
import torch

from .config import EvalConfig, MAMBA2_MODELS, PROJECT_ROOT
from .data import load_bioasq_eval, load_pubmedqa_eval
from .evaluate import run_evaluation


def main():
    p = argparse.ArgumentParser(description="BioMamba Evaluation")
    p.add_argument("--model_path", required=True, help="Path to fine-tuned model")
    p.add_argument("--base_model", default=None,
                   help="Base model for comparison (e.g., state-spaces/mamba2-130m)")
    p.add_argument("--dataset", default="bioasq", choices=["bioasq", "pubmedqa"])
    p.add_argument("--data_path", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--output_dir", default=os.path.join(PROJECT_ROOT, "evaluation_results"))
    args = p.parse_args()

    cfg = EvalConfig(
        model_path=args.model_path,
        base_model_path=args.base_model,
        dataset_name=args.dataset,
        data_path=args.data_path or os.path.join(PROJECT_ROOT, "data", "bioasq_test"),
        split=args.split,
        max_length=args.max_length,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        gpu_id=args.gpu_id,
        output_dir=args.output_dir,
    )

    if args.dataset == "bioasq":
        ds = load_bioasq_eval(cfg.data_path, cfg.split)
    else:
        ds = load_pubmedqa_eval(cfg.split)

    print(f"Evaluation dataset: {args.dataset}, samples: {len(ds)}")
    run_evaluation(cfg, ds)


if __name__ == "__main__":
    main()
