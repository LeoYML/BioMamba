"""
Unified CLI entry point for MIMIC clinical evaluation tasks.

Usage:
    # Perplexity
    python -m ft_biomamba.run_mimic_eval \
        --model_path ./checkpoints/mixed_wiki/biomamba_cpt_singledoc_mamba2-130m/best_model \
        --task perplexity --mimic_dir ./data/mimic-iv-note

    # Completion (ROUGE)
    python -m ft_biomamba.run_mimic_eval \
        --model_path ... --task completion --max_new_tokens 128

    # Mortality prediction
    python -m ft_biomamba.run_mimic_eval \
        --model_path ... --task mortality \
        --admissions_path ./data/mimic-iv/hosp/admissions.csv.gz

    # Discharge summary generation
    python -m ft_biomamba.run_mimic_eval \
        --model_path ... --task discharge --max_new_tokens 256
"""

import argparse
import sys

import torch
from torch.utils.data import DataLoader

from .config import PROJECT_ROOT
from .model import load_model, load_tokenizer
from .mimic_config import MIMICEvalConfig
from .mimic_data import (
    load_mimic_notes,
    split_mimic_data,
    prepare_perplexity_data,
    prepare_completion_data,
    prepare_mortality_data,
    prepare_discharge_data,
)
from .mimic_evaluate import (
    evaluate_clinical_ppl,
    evaluate_completion,
    evaluate_mortality,
    evaluate_discharge,
    save_results,
    print_ppl_results,
    print_generation_results,
)


def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    return {"input_ids": input_ids}


def parse_args():
    p = argparse.ArgumentParser(description="BioMamba MIMIC clinical evaluation")

    # Model
    p.add_argument("--model_path", required=True, help="Path to model checkpoint")
    p.add_argument("--model_type", default="mamba2", choices=["mamba2", "t5", "auto"])

    # MIMIC data
    p.add_argument("--mimic_dir", default=os.path.join(PROJECT_ROOT, "data/mimic-iv-note/2.2/note"),
                    help="Directory containing MIMIC note CSV files")
    p.add_argument("--mimic_version", default="iv", choices=["iii", "iv"])
    p.add_argument("--admissions_path", default=None,
                    help="Path to admissions.csv.gz (required for mortality task)")
    p.add_argument("--note_category", default=None,
                    help="Note category filter (e.g. 'discharge', 'radiology')")

    # Task
    p.add_argument("--task", required=True,
                    choices=["perplexity", "completion", "mortality", "discharge"],
                    help="Evaluation task")

    # Context handling
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--chunk_stride", type=int, default=512)
    p.add_argument("--max_context_chars", type=int, default=4000)

    # Generation
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)

    # Evaluation
    p.add_argument("--max_samples", type=int, default=500)
    p.add_argument("--min_note_length", type=int, default=200)
    p.add_argument("--prefix_ratio", type=float, default=0.5)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)

    # Output
    p.add_argument("--output_dir", default=os.path.join(PROJECT_ROOT, "evaluation_results/mimic"))
    p.add_argument("--no_save_predictions", action="store_true")

    # Hardware
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=16)

    return p.parse_args()


def main():
    args = parse_args()

    # Build config
    cfg = MIMICEvalConfig(
        model_path=args.model_path,
        model_type=args.model_type,
        mimic_version=args.mimic_version,
        mimic_data_dir=args.mimic_dir,
        admissions_path=args.admissions_path,
        task=args.task,
        max_length=args.max_length,
        context_strategy="sliding_window",
        chunk_stride=args.chunk_stride,
        max_context_chars=args.max_context_chars,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        max_samples=args.max_samples,
        note_category=args.note_category,
        min_note_length=args.min_note_length,
        prefix_ratio=args.prefix_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        output_dir=args.output_dir,
        save_predictions=not args.no_save_predictions,
        gpu_id=args.gpu_id,
        batch_size=args.batch_size,
    )

    # Device
    if torch.cuda.is_available():
        device = f"cuda:{cfg.gpu_id}"
        torch.cuda.set_device(cfg.gpu_id)
    else:
        device = "cpu"

    print(f"\n{'='*60}")
    print(f"  MIMIC {cfg.task.upper()} Evaluation")
    print(f"  Model: {cfg.model_path}")
    print(f"  MIMIC: {cfg.mimic_data_dir} (v{cfg.mimic_version})")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # Load MIMIC notes
    category = cfg.note_category
    if category is None and cfg.task != "mortality":
        category = "discharge" if cfg.mimic_version == "iv" else "Discharge summary"

    notes_df = load_mimic_notes(
        cfg.mimic_data_dir,
        version=cfg.mimic_version,
        category=category,
        min_length=cfg.min_note_length,
    )
    _, test_df = split_mimic_data(notes_df, test_ratio=cfg.test_ratio, seed=cfg.seed)

    # Load model
    print(f"\nLoading model: {cfg.model_path}")
    model = load_model(cfg.model_path, device=device)
    model.eval()
    tokenizer = load_tokenizer()

    # Dispatch by task
    if cfg.task == "perplexity":
        dataset = prepare_perplexity_data(
            test_df, tokenizer,
            max_length=cfg.max_length,
            stride=cfg.chunk_stride,
            max_notes=cfg.max_samples,
        )
        loader = DataLoader(
            dataset, batch_size=cfg.batch_size, shuffle=False,
            num_workers=2, pin_memory=True, collate_fn=collate_fn,
        )
        loss, ppl = evaluate_clinical_ppl(model, loader, device)
        print_ppl_results(cfg.model_path, loss, ppl)
        save_results("perplexity", {"loss": loss, "ppl": ppl}, [], cfg)

    elif cfg.task == "completion":
        samples = prepare_completion_data(
            test_df, tokenizer,
            prefix_ratio=cfg.prefix_ratio,
            max_length=cfg.max_length,
            max_samples=cfg.max_samples,
        )
        metrics, predictions = evaluate_completion(model, tokenizer, samples, cfg, device)
        print_generation_results("completion", metrics)
        save_results("completion", metrics, predictions, cfg)

    elif cfg.task == "mortality":
        if not cfg.admissions_path:
            print("ERROR: --admissions_path required for mortality task")
            print("  e.g. --admissions_path ./data/mimic-iv/hosp/admissions.csv.gz")
            sys.exit(1)
        samples = prepare_mortality_data(
            test_df, cfg.admissions_path,
            version=cfg.mimic_version,
            max_context_chars=cfg.max_context_chars,
            max_samples=cfg.max_samples,
        )
        metrics, predictions = evaluate_mortality(model, tokenizer, samples, cfg, device)
        print_generation_results("mortality", metrics)
        save_results("mortality", metrics, predictions, cfg)

    elif cfg.task == "discharge":
        samples = prepare_discharge_data(
            test_df,
            max_context_chars=cfg.max_context_chars,
            max_samples=cfg.max_samples,
        )
        metrics, predictions = evaluate_discharge(model, tokenizer, samples, cfg, device)
        print_generation_results("discharge", metrics)
        save_results("discharge", metrics, predictions, cfg)

    else:
        print(f"Unknown task: {cfg.task}")
        sys.exit(1)

    # Cleanup
    del model
    torch.cuda.empty_cache()
    print("\nDone.")


if __name__ == "__main__":
    main()
