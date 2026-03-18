#!/usr/bin/env python3
"""Generate more case study samples, focusing on finding examples where CPT+SFT wins."""
# --- Project root resolution (auto-generated) ---
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')
_sys.path.insert(0, _PROJECT_ROOT)
_os.chdir(_PROJECT_ROOT)
# --- End project root resolution ---

import json, os, sys, torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from ft_biomamba.model import load_model, load_tokenizer
from ft_biomamba.mimic_data import load_mimic_notes, split_mimic_data, prepare_completion_data, prepare_discharge_data
from ft_biomamba.mimic_prompts import format_completion_prompt, format_discharge_prompt
from ft_biomamba.mimic_evaluate import _generate_raw
from rouge_score import rouge_scorer

MODELS = {
    "base": "state-spaces/mamba2-1.3b",
    "base+SFT": _os.path.join(_PROJECT_ROOT, "checkpoints/mimic_sft_1.3b/biomamba_mimic_sft_mamba2-1.3b_both_full/best_model"),
    "CPT+SFT": _os.path.join(_PROJECT_ROOT, "checkpoints/mimic_sft_1.3b_cpt_v5/biomamba_mimic_sft_mamba2-1.3b_both_full/best_model"),
}
DEVICE = "cuda"

def main():
    tokenizer = load_tokenizer()
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    print("Loading MIMIC data...")
    notes = load_mimic_notes(_os.path.join(_PROJECT_ROOT, "data/mimic-iv-note/2.2/note"))
    _, test_df = split_mimic_data(notes, test_ratio=0.1, seed=42)
    comp_samples = prepare_completion_data(test_df, tokenizer, max_samples=500)
    disc_samples = prepare_discharge_data(test_df, max_samples=500)

    # Use indices 10-30 for more diversity
    comp_indices = list(range(10, 30))
    disc_indices = list(range(10, 30))

    all_results = {"completion": [], "discharge": []}

    # Load all models
    models = {}
    for name, path in MODELS.items():
        print(f"Loading {name}...")
        models[name] = load_model(path, device=DEVICE)
        models[name].eval()

    # Generate completion samples
    print("\n=== COMPLETION ===")
    for ci, idx in enumerate(comp_indices):
        sample = comp_samples[idx]
        prefix = sample["prefix_text"]
        reference = sample["reference_text"]
        entry = {"idx": idx, "prefix": prefix[-600:], "reference": reference[:600], "generations": {}, "scores": {}}

        for mname, model in models.items():
            prompt = format_completion_prompt(prefix)
            gen = _generate_raw(model, tokenizer, prompt, torch.device(DEVICE),
                                max_new_tokens=128, temperature=0.0, max_length=1024)
            sc = scorer.score(reference, gen) if gen and reference else None
            entry["generations"][mname] = gen[:600]
            entry["scores"][mname] = {
                "rouge1": round(sc["rouge1"].fmeasure * 100, 2) if sc else 0,
                "rouge2": round(sc["rouge2"].fmeasure * 100, 2) if sc else 0,
                "rougeL": round(sc["rougeL"].fmeasure * 100, 2) if sc else 0,
            }

        # Check if CPT+SFT beats base+SFT
        cpt_r1 = entry["scores"]["CPT+SFT"]["rouge1"]
        sft_r1 = entry["scores"]["base+SFT"]["rouge1"]
        base_r1 = entry["scores"]["base"]["rouge1"]
        marker = "***" if cpt_r1 > sft_r1 else ""
        print(f"  Comp [{idx}]: base={base_r1:.1f} sft={sft_r1:.1f} cpt+sft={cpt_r1:.1f} {marker}")
        all_results["completion"].append(entry)

    # Generate discharge samples
    print("\n=== DISCHARGE ===")
    for di, idx in enumerate(disc_indices):
        sample = disc_samples[idx]
        entry = {"idx": idx, "context": sample["context"][:600], "reference": sample["reference"][:600],
                 "generations": {}, "scores": {}}

        for mname, model in models.items():
            prompt = format_discharge_prompt(sample["context"], max_ctx_chars=4000)
            gen = _generate_raw(model, tokenizer, prompt, torch.device(DEVICE),
                                max_new_tokens=128, temperature=0.0, max_length=1024)
            sc = scorer.score(sample["reference"], gen) if gen and sample["reference"] else None
            entry["generations"][mname] = gen[:600]
            entry["scores"][mname] = {
                "rouge1": round(sc["rouge1"].fmeasure * 100, 2) if sc else 0,
                "rouge2": round(sc["rouge2"].fmeasure * 100, 2) if sc else 0,
                "rougeL": round(sc["rougeL"].fmeasure * 100, 2) if sc else 0,
            }

        cpt_r1 = entry["scores"]["CPT+SFT"]["rouge1"]
        sft_r1 = entry["scores"]["base+SFT"]["rouge1"]
        base_r1 = entry["scores"]["base"]["rouge1"]
        marker = "***" if cpt_r1 > sft_r1 else ""
        print(f"  Disc [{idx}]: base={base_r1:.1f} sft={sft_r1:.1f} cpt+sft={cpt_r1:.1f} {marker}")
        all_results["discharge"].append(entry)

    # Cleanup
    for m in models.values():
        del m
    torch.cuda.empty_cache()

    out_path = _os.path.join(_PROJECT_ROOT, "evaluation_results/case_study_1.3b_v2.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
