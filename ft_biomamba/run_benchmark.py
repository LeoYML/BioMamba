#!/usr/bin/env python3
"""
Benchmark multiple models on BioASQ.

Usage:
  python -m ft_biomamba.run_benchmark                          # all default models
  python -m ft_biomamba.run_benchmark --models flan-t5-large biomamba-sft
  python -m ft_biomamba.run_benchmark --extra my-model:/path/to/model:t5
"""

import argparse
import os
from collections import OrderedDict

from .config import EvalConfig, MAMBA2_MODELS, BENCHMARK_MODELS, PROJECT_ROOT
from .data import load_bioasq_eval, load_pubmedqa_eval, load_medqa_eval
from .evaluate import run_benchmark


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark models on BioASQ")

    p.add_argument("--models", nargs="*", default=None,
                   help="Model names to evaluate (from registry). Default: all")
    p.add_argument("--extra", nargs="*", default=[],
                   help="Extra models as name:path:type (e.g. my-model:/path:t5)")

    # Mamba checkpoints
    p.add_argument("--biomamba_sft", default=None,
                   help="Path to BioMamba SFT checkpoint")
    p.add_argument("--biomamba_sft_nocpt", default=None,
                   help="Path to BioMamba SFT (no CPT) checkpoint")
    p.add_argument("--biomamba_base_fewshot", action="store_true", default=True,
                   help="Include base mamba2-130m with few-shot prompt")
    p.add_argument("--no_biomamba_base", dest="biomamba_base_fewshot", action="store_false")

    # Data
    p.add_argument("--dataset", default="bioasq", choices=["bioasq", "pubmedqa", "medqa"])
    p.add_argument("--data_path", default=None)
    p.add_argument("--max_samples", type=int, default=None)

    # Hardware
    p.add_argument("--gpu_id", type=int, default=0)

    # Output
    p.add_argument("--output_dir", default=os.path.join(PROJECT_ROOT, "evaluation_results"))

    return p.parse_args()


def main():
    args = parse_args()

    # Build model list
    models = OrderedDict()

    # Add Mamba models
    if args.biomamba_sft:
        models["BioMamba-CPT+SFT"] = {"path": args.biomamba_sft, "type": "mamba2"}
    elif os.path.exists(os.path.join(PROJECT_ROOT, "checkpoints/biomamba_sft_mamba2-130m_full/best_model")):
        models["BioMamba-CPT+SFT"] = {
            "path": os.path.join(PROJECT_ROOT, "checkpoints/biomamba_sft_mamba2-130m_full/best_model"), "type": "mamba2"
        }

    if args.biomamba_sft_nocpt:
        models["BioMamba-SFT-only"] = {"path": args.biomamba_sft_nocpt, "type": "mamba2"}
    elif os.path.exists(os.path.join(PROJECT_ROOT, "checkpoints/nocpt/biomamba_sft_mamba2-130m_full/best_model")):
        models["BioMamba-SFT-only"] = {
            "path": os.path.join(PROJECT_ROOT, "checkpoints/nocpt/biomamba_sft_mamba2-130m_full/best_model"), "type": "mamba2"
        }

    if args.biomamba_base_fewshot:
        models["Mamba2-130m-base"] = {"path": "state-spaces/mamba2-130m", "type": "mamba2", "fewshot": True}

    # Add T5 / benchmark models
    if args.models is None:
        # Default: all registered benchmark models
        for name, info in BENCHMARK_MODELS.items():
            models[name] = info.copy()
    else:
        for name in args.models:
            if name in BENCHMARK_MODELS:
                models[name] = BENCHMARK_MODELS[name].copy()
            elif name in MAMBA2_MODELS:
                models[name] = {"path": MAMBA2_MODELS[name]["path"], "type": "mamba2"}
            else:
                print(f"[warn] Unknown model: {name}, skipping")

    # Add extra models
    for spec in args.extra:
        parts = spec.split(":")
        if len(parts) >= 2:
            name = parts[0]
            path = parts[1]
            mtype = parts[2] if len(parts) > 2 else "auto"
            models[name] = {"path": path, "type": mtype}

    if not models:
        print("No models to evaluate!")
        return

    print(f"Will benchmark {len(models)} models:")
    for name, info in models.items():
        tag = " (few-shot)" if info.get("fewshot") else ""
        print(f"  - {name}: {info['path']} [{info['type']}]{tag}")

    # Load dataset
    data_path = args.data_path or os.path.join(PROJECT_ROOT, "data", "bioasq_test")
    is_mcq = args.dataset == "medqa"

    if args.dataset == "bioasq":
        dataset = load_bioasq_eval(data_path, "test")
    elif args.dataset == "pubmedqa":
        dataset = load_pubmedqa_eval("test", 42)
    elif args.dataset == "medqa":
        dataset = load_medqa_eval("test", args.max_samples or 200)
    print(f"\nDataset: {args.dataset}  Samples: {len(dataset)}")

    # Config
    cfg = EvalConfig(
        dataset_name=args.dataset,
        data_path=data_path,
        max_samples=args.max_samples,
        gpu_id=args.gpu_id,
        output_dir=args.output_dir,
    )

    # For MCQ datasets, use logprob-based evaluation for all models
    if is_mcq:
        import torch
        from .evaluate import load_eval_model, evaluate_mcq, _print_benchmark
        device = torch.device("cuda", cfg.gpu_id) if torch.cuda.is_available() else torch.device("cpu")
        all_results = {}
        for name, info in models.items():
            print(f"\n{'='*65}")
            print(f"  Evaluating: {name}")
            print(f"{'='*65}")
            model, tokenizer, detected = load_eval_model(
                info["path"], info.get("type", "auto"), str(device)
            )
            metrics, _ = evaluate_mcq(model, tokenizer, dataset, cfg, device,
                                      model_type=detected, tag=name)
            metrics["model_path"] = info["path"]
            metrics["model_type"] = detected
            metrics["params"] = sum(p.numel() for p in model.parameters())
            all_results[name] = metrics
            del model
            torch.cuda.empty_cache()

        _print_benchmark(all_results, cfg)

        # Save
        import json
        from datetime import datetime
        os.makedirs(cfg.output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(cfg.output_dir, f"benchmark_{cfg.dataset_name}_{ts}.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nBenchmark results saved to {out_path}")
        return

    # Handle few-shot models: we need to override prompt selection for base models.
    fewshot_models = {k: v for k, v in models.items() if v.get("fewshot")}
    regular_models = {k: v for k, v in models.items() if not v.get("fewshot")}

    all_results = {}

    if regular_models:
        results = run_benchmark(regular_models, dataset, cfg)
        all_results.update(results)

    # Evaluate few-shot models separately
    if fewshot_models:
        import torch
        from .evaluate import load_eval_model, evaluate_model_on_dataset
        device = torch.device("cuda", cfg.gpu_id) if torch.cuda.is_available() else torch.device("cpu")
        for name, info in fewshot_models.items():
            print(f"\n{'='*65}")
            print(f"  Evaluating: {name} (few-shot prompt)")
            print(f"{'='*65}")
            model, tokenizer, detected = load_eval_model(info["path"], info["type"], str(device))
            metrics, _ = evaluate_model_on_dataset(
                model, tokenizer, dataset, cfg, device, tag=name,
                use_fewshot=True, model_type=detected
            )
            metrics["model_path"] = info["path"]
            metrics["model_type"] = detected
            metrics["params"] = sum(p.numel() for p in model.parameters())
            all_results[name] = metrics
            del model
            torch.cuda.empty_cache()

    # Final combined table
    if all_results:
        from .evaluate import _print_benchmark
        print("\n\n")
        _print_benchmark(all_results, cfg)


if __name__ == "__main__":
    main()
