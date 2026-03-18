#!/usr/bin/env python3
"""Generate case study samples from 1.3b base / base+SFT / CPT+SFT models.

Picks a few representative samples from completion and discharge tasks,
generates text from all three models, and saves results as JSON for MD writing.
"""
# --- Project root resolution (auto-generated) ---
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')
_sys.path.insert(0, _PROJECT_ROOT)
_os.chdir(_PROJECT_ROOT)
# --- End project root resolution ---

import json
import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from ft_biomamba.model import load_model, load_tokenizer
from ft_biomamba.mimic_data import load_mimic_notes, split_mimic_data, prepare_completion_data, prepare_discharge_data
from ft_biomamba.mimic_prompts import format_completion_prompt, format_discharge_prompt
from ft_biomamba.mimic_evaluate import _generate_raw
from rouge_score import rouge_scorer

# --- Config ---
MODELS = {
    "base": "state-spaces/mamba2-1.3b",
    "base+SFT": os.path.join(_PROJECT_ROOT, "checkpoints/mimic_sft_1.3b/biomamba_mimic_sft_mamba2-1.3b_both_full/best_model"),
    "CPT+SFT": os.path.join(_PROJECT_ROOT, "checkpoints/mimic_sft_1.3b_cpt_v5/biomamba_mimic_sft_mamba2-1.3b_both_full/best_model"),
}
NUM_SAMPLES = 5  # pick 5, then choose best examples
DEVICE = "cuda"
MAX_NEW_TOKENS = 128
SEED = 42

def main():
    tokenizer = load_tokenizer()
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    # Load test data
    print("Loading MIMIC data...")
    notes = load_mimic_notes(os.path.join(_PROJECT_ROOT, "data/mimic-iv-note/2.2/note"))
    _, test_df = split_mimic_data(notes, test_ratio=0.1, seed=42)

    comp_samples = prepare_completion_data(test_df, tokenizer, max_samples=500)
    disc_samples = prepare_discharge_data(test_df, max_samples=500)

    # Pick diverse samples with enough content
    rng = np.random.RandomState(SEED)
    comp_indices = rng.choice(len(comp_samples), min(NUM_SAMPLES, len(comp_samples)), replace=False)
    disc_indices = rng.choice(len(disc_samples), min(NUM_SAMPLES, len(disc_samples)), replace=False)

    results = {"completion": [], "discharge": []}

    for model_name, model_path in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Loading model: {model_name} ({model_path})")
        model = load_model(model_path, device=DEVICE)
        model.eval()

        # --- Completion ---
        print(f"\nGenerating completion samples for {model_name}...")
        for ci, idx in enumerate(comp_indices):
            sample = comp_samples[idx]
            prompt = format_completion_prompt(sample["prefix_text"])
            generated = _generate_raw(
                model, tokenizer, prompt, torch.device(DEVICE),
                max_new_tokens=MAX_NEW_TOKENS, temperature=0.0, max_length=1024,
            )
            reference = sample["reference_text"]
            scores = scorer.score(reference, generated) if generated and reference else None

            if model_name == list(MODELS.keys())[0]:
                # First model: create entry
                results["completion"].append({
                    "idx": int(idx),
                    "prefix": sample["prefix_text"][-500:],  # last 500 chars for context
                    "reference": reference[:500],
                    "generations": {},
                    "scores": {},
                })

            entry = results["completion"][ci]
            entry["generations"][model_name] = generated[:500]
            entry["scores"][model_name] = {
                "rouge1": round(scores["rouge1"].fmeasure * 100, 2) if scores else 0,
                "rouge2": round(scores["rouge2"].fmeasure * 100, 2) if scores else 0,
                "rougeL": round(scores["rougeL"].fmeasure * 100, 2) if scores else 0,
            }
            print(f"  Sample {ci}: R1={entry['scores'][model_name]['rouge1']:.1f}")

        # --- Discharge ---
        print(f"\nGenerating discharge samples for {model_name}...")
        for di, idx in enumerate(disc_indices):
            sample = disc_samples[idx]
            prompt = format_discharge_prompt(sample["context"], max_ctx_chars=4000)
            generated = _generate_raw(
                model, tokenizer, prompt, torch.device(DEVICE),
                max_new_tokens=MAX_NEW_TOKENS, temperature=0.0, max_length=1024,
            )
            reference = sample["reference"]
            scores = scorer.score(reference, generated) if generated and reference else None

            if model_name == list(MODELS.keys())[0]:
                results["discharge"].append({
                    "idx": int(idx),
                    "context": sample["context"][:500],
                    "reference": reference[:500],
                    "generations": {},
                    "scores": {},
                })

            entry = results["discharge"][di]
            entry["generations"][model_name] = generated[:500]
            entry["scores"][model_name] = {
                "rouge1": round(scores["rouge1"].fmeasure * 100, 2) if scores else 0,
                "rouge2": round(scores["rouge2"].fmeasure * 100, 2) if scores else 0,
                "rougeL": round(scores["rougeL"].fmeasure * 100, 2) if scores else 0,
            }
            print(f"  Sample {di}: R1={entry['scores'][model_name]['rouge1']:.1f}")

        # Free memory
        del model
        torch.cuda.empty_cache()

    # Save results
    out_path = os.path.join(_PROJECT_ROOT, "evaluation_results/case_study_1.3b.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
