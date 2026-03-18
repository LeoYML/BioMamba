"""
Evaluate base vs CPT model PPL on MIMIC clinical text domains.

Compares 130m base vs CPT on:
  - MIMIC-IV discharge summaries
  - MIMIC-IV radiology reports (if available)

Shows whether CPT on PubMed + Wikipedia improves clinical text understanding.

Usage:
    python eval_mimic_ppl.py [--mimic_dir ./data/mimic-iv-note] [--max_notes 500]
"""

# --- Project root resolution (auto-generated) ---
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')
_sys.path.insert(0, _PROJECT_ROOT)
_os.chdir(_PROJECT_ROOT)
# --- End project root resolution ---

import argparse
import json
import math
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ft_biomamba.model import load_model, load_tokenizer
from ft_biomamba.mimic_data import (
    load_mimic_notes,
    split_mimic_data,
    prepare_perplexity_data,
)
from ft_biomamba.mimic_evaluate import evaluate_clinical_ppl, print_ppl_results

DEVICE = "cuda:0"
BATCH = 16
SEQ_LEN = 1024
STRIDE = 512

RESULTS_DIR = os.path.join(_PROJECT_ROOT, "evaluation_results/mimic")
RESULTS_JSON = os.path.join(RESULTS_DIR, "mimic_ppl_comparison.json")

# Models to evaluate: (display_name, model_path)
MODELS = [
    ("130m-base", "state-spaces/mamba2-130m"),
    ("130m-CPT", os.path.join(_PROJECT_ROOT, "checkpoints/mixed_wiki/biomamba_cpt_singledoc_mamba2-130m/best_model")),
]


def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    return {"input_ids": input_ids}


def build_mimic_dataset(tokenizer, mimic_dir, category, version="iv",
                         max_notes=500, seq_len=SEQ_LEN, stride=STRIDE):
    """Load MIMIC notes and prepare tokenized chunks for PPL evaluation."""
    try:
        notes_df = load_mimic_notes(mimic_dir, version=version, category=category)
    except FileNotFoundError as e:
        print(f"  SKIP: {e}")
        return None

    _, test_df = split_mimic_data(notes_df, test_ratio=0.1, seed=42)
    dataset = prepare_perplexity_data(
        test_df, tokenizer, max_length=seq_len, stride=stride, max_notes=max_notes
    )
    return dataset


def main():
    parser = argparse.ArgumentParser(description="MIMIC clinical PPL comparison")
    parser.add_argument("--mimic_dir", default=os.path.join(_PROJECT_ROOT, "data/mimic-iv-note/2.2/note"),
                        help="Directory containing MIMIC note CSV files")
    parser.add_argument("--mimic_version", default="iv", choices=["iii", "iv"])
    parser.add_argument("--max_notes", type=int, default=500,
                        help="Max notes per domain for evaluation")
    parser.add_argument("--batch_size", type=int, default=BATCH)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer()

    # Define domains based on MIMIC version
    if args.mimic_version == "iv":
        domains = {
            "discharge": "discharge",
            "radiology": "radiology",
        }
    else:
        domains = {
            "discharge": "Discharge summary",
            "radiology": "Radiology",
            "nursing": "Nursing",
        }

    # Build datasets
    datasets = {}
    for domain_name, category in domains.items():
        print(f"\n--- Building {domain_name} dataset ---")
        ds = build_mimic_dataset(
            tokenizer, args.mimic_dir, category,
            version=args.mimic_version, max_notes=args.max_notes,
        )
        if ds is not None:
            datasets[domain_name] = ds

    if not datasets:
        print("\nNo MIMIC datasets found. Check --mimic_dir path.")
        print(f"Expected files in: {args.mimic_dir}")
        if args.mimic_version == "iv":
            print("  - discharge.csv.gz")
            print("  - radiology.csv.gz (optional)")
        else:
            print("  - NOTEEVENTS.csv.gz")
        sys.exit(1)

    # Build data loaders
    loaders = {}
    for name, ds in datasets.items():
        loaders[name] = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=2, pin_memory=True, collate_fn=collate_fn,
        )

    # Evaluate each model
    results = {}
    for model_name, model_path in MODELS:
        print(f"\n{'='*60}")
        print(f"  {model_name}  —  {model_path}")
        print(f"{'='*60}")

        try:
            model = load_model(model_path, device=device)
        except Exception as e:
            print(f"  ERROR loading model: {e}")
            results[model_name] = {d: None for d in datasets}
            continue

        model_results = {}
        for domain_name, loader in loaders.items():
            print(f"\n  [{domain_name}]")
            loss, ppl = evaluate_clinical_ppl(model, loader, device)
            print_ppl_results(f"{model_name} / {domain_name}", loss, ppl)
            model_results[domain_name] = {"loss": loss, "ppl": ppl}

        results[model_name] = model_results
        del model
        torch.cuda.empty_cache()

        # Incremental save
        _save(results)

    # Summary table
    _print_summary(results, datasets.keys())
    _save(results)


def _save(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [saved → {RESULTS_JSON}]")

    # TSV
    tsv_path = os.path.join(RESULTS_DIR, "mimic_ppl_comparison.tsv")
    with open(tsv_path, "w") as f:
        domains = set()
        for r in results.values():
            if r:
                domains.update(r.keys())
        domains = sorted(domains)

        header = "model\t" + "\t".join(f"{d}_ppl\t{d}_loss" for d in domains)
        f.write(header + "\n")

        for name, r in results.items():
            parts = [name]
            for d in domains:
                if r and r.get(d) and r[d].get("ppl") is not None:
                    parts.append(f"{r[d]['ppl']:.4f}")
                    parts.append(f"{r[d]['loss']:.4f}")
                else:
                    parts.extend(["ERROR", "ERROR"])
            f.write("\t".join(parts) + "\n")


def _print_summary(results, domain_names):
    domain_names = list(domain_names)

    print(f"\n{'='*70}")
    header = f"{'Model':<15}" + "".join(f" {d:>15}" for d in domain_names)
    print(header)
    print("-" * 70)

    for name, r in results.items():
        parts = [f"{name:<15}"]
        for d in domain_names:
            if r and r.get(d) and r[d].get("ppl") is not None:
                parts.append(f" {r[d]['ppl']:>14.2f}")
            else:
                parts.append(f" {'ERROR':>14}")
        print("".join(parts))

    # Delta table (base → CPT)
    base_names = [n for n in results if "base" in n.lower()]
    cpt_names = [n for n in results if "cpt" in n.lower()]

    if base_names and cpt_names:
        print(f"\n{'='*80}")
        print(f"{'Size':<8} {'Domain':<15} {'Base PPL':>10} {'CPT PPL':>10} {'Δ PPL':>8} {'Improv%':>8}")
        print("-" * 80)

        for base_name in base_names:
            size = base_name.split("-")[0]
            cpt_name = f"{size}-CPT"
            if cpt_name not in results:
                continue

            for d in domain_names:
                br = results[base_name].get(d)
                cr = results[cpt_name].get(d)
                if br and cr and br.get("ppl") and cr.get("ppl"):
                    bp, cp = br["ppl"], cr["ppl"]
                    delta = bp - cp
                    pct = delta / bp * 100
                    print(f"{size:<8} {d:<15} {bp:>10.2f} {cp:>10.2f} {delta:>8.2f} {pct:>7.1f}%")

        print("=" * 80)


if __name__ == "__main__":
    main()
