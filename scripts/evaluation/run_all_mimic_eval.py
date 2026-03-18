#!/usr/bin/env python3
"""Batch MIMIC evaluation: all models × all tasks × 500 samples."""

# --- Project root resolution (auto-generated) ---
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')
_sys.path.insert(0, _PROJECT_ROOT)
_os.chdir(_PROJECT_ROOT)
# --- End project root resolution ---

import json
import os
import sys
import gc
import torch

# Models to evaluate
MODELS = [
    ("130m-base", "state-spaces/mamba2-130m"),
    ("130m-CPT", os.path.join(_PROJECT_ROOT, "checkpoints/mixed_wiki/biomamba_cpt_singledoc_mamba2-130m/best_model")),
    ("130m-CPT+SFT", os.path.join(_PROJECT_ROOT, "checkpoints/mimic_sft_v2/biomamba_mimic_sft_mamba2-130m_both_full/best_model")),
    ("130m-base+SFT", os.path.join(_PROJECT_ROOT, "checkpoints/mimic_sft_nocpt_v2/biomamba_mimic_sft_mamba2-130m_both_full/best_model")),
]

# 130m CPT+SFT v3
_130m_v3 = os.path.join(_PROJECT_ROOT, "checkpoints/mimic_sft_130m_cpt_v3/biomamba_mimic_sft_mamba2-130m_both_full/best_model")
if os.path.exists(_130m_v3):
    MODELS.append(("130m-CPT+SFT-v3", _130m_v3))

# Add larger models if their SFT checkpoints exist
for size, hf_name in [("370m", "state-spaces/mamba2-370m"), ("780m", "state-spaces/mamba2-780m"),
                       ("1.3b", "state-spaces/mamba2-1.3b"), ("2.7b", "state-spaces/mamba2-2.7b")]:
    # Base model
    MODELS.append((f"{size}-base", hf_name))

    # CPT (PubMed+Wiki) model
    cpt_path = os.path.join(_PROJECT_ROOT, f"checkpoints/{size}_mixed_wiki/biomamba_cpt_singledoc_mamba2-{size}/best_model")
    if os.path.exists(cpt_path):
        MODELS.append((f"{size}-CPT", cpt_path))

    # Base+SFT model
    base_sft_path = os.path.join(_PROJECT_ROOT, f"checkpoints/mimic_sft_{size}/biomamba_mimic_sft_mamba2-{size}_both_full/best_model")
    if os.path.exists(base_sft_path):
        MODELS.append((f"{size}-base+SFT", base_sft_path))

    # CPT+SFT model (v1)
    cpt_sft_path = os.path.join(_PROJECT_ROOT, f"checkpoints/mimic_sft_{size}_cpt/biomamba_mimic_sft_mamba2-{size}_both_full/best_model")
    if os.path.exists(cpt_sft_path):
        MODELS.append((f"{size}-CPT+SFT", cpt_sft_path))

    # CPT+SFT v2 (retuned with lower lr)
    cpt_v2_path = os.path.join(_PROJECT_ROOT, f"checkpoints/mimic_sft_{size}_cpt_v2/biomamba_mimic_sft_mamba2-{size}_both_full/best_model")
    if os.path.exists(cpt_v2_path):
        MODELS.append((f"{size}-CPT+SFT-v2", cpt_v2_path))

    # CPT+SFT v3 (final tuned with more data)
    for v3_dir_name in [f"checkpoints/mimic_sft_{size}_cpt_v3"]:
        cpt_v3_path = os.path.join(_PROJECT_ROOT, f"{v3_dir_name}/biomamba_mimic_sft_mamba2-{size}_both_full/best_model")
        if os.path.exists(cpt_v3_path):
            MODELS.append((f"{size}-CPT+SFT-v3", cpt_v3_path))
            break

    # CPT+SFT v4 (targeted fix with even lower lr)
    cpt_v4_path = os.path.join(_PROJECT_ROOT, f"checkpoints/mimic_sft_{size}_cpt_v4/biomamba_mimic_sft_mamba2-{size}_both_full/best_model")
    if os.path.exists(cpt_v4_path):
        MODELS.append((f"{size}-CPT+SFT-v4", cpt_v4_path))

    # CPT+SFT v5 (balanced lr for 780m)
    cpt_v5_path = os.path.join(_PROJECT_ROOT, f"checkpoints/mimic_sft_{size}_cpt_v5/biomamba_mimic_sft_mamba2-{size}_both_full/best_model")
    if os.path.exists(cpt_v5_path):
        MODELS.append((f"{size}-CPT+SFT-v5", cpt_v5_path))

    # Base+SFT v2 (retrained with lower lr)
    v2_path = os.path.join(_PROJECT_ROOT, f"checkpoints/mimic_sft_{size}_v2/biomamba_mimic_sft_mamba2-{size}_both_full/best_model")
    if os.path.exists(v2_path):
        MODELS.append((f"{size}-base+SFT-v2", v2_path))

MAX_SAMPLES = 500
MIMIC_DIR = os.path.join(_PROJECT_ROOT, "data/mimic-iv-note/2.2/note")
OUTPUT = os.path.join(_PROJECT_ROOT, "evaluation_results/mimic_v2")

def main():
    from ft_biomamba.model import load_model, load_tokenizer
    from ft_biomamba.mimic_config import MIMICEvalConfig
    from ft_biomamba.mimic_data import (
        load_mimic_notes, split_mimic_data,
        prepare_perplexity_data, prepare_completion_data, prepare_discharge_data,
    )
    from ft_biomamba.mimic_evaluate import (
        evaluate_clinical_ppl, evaluate_completion, evaluate_discharge,
        save_results, print_ppl_results, print_generation_results,
    )
    from torch.utils.data import DataLoader

    os.makedirs(OUTPUT, exist_ok=True)
    tokenizer = load_tokenizer()

    # Load MIMIC data once
    print("Loading MIMIC data...")
    notes_df = load_mimic_notes(MIMIC_DIR, version="iv", category="discharge")
    _, test_df = split_mimic_data(notes_df, test_ratio=0.1, seed=42)

    # Prepare eval data
    print("Preparing eval data...")
    ppl_dataset = prepare_perplexity_data(test_df, tokenizer, max_length=1024, stride=512, max_notes=500)
    ppl_loader = DataLoader(ppl_dataset, batch_size=16, shuffle=False)

    completion_samples = prepare_completion_data(test_df, tokenizer, max_samples=MAX_SAMPLES)
    discharge_samples = prepare_discharge_data(test_df, max_samples=MAX_SAMPLES)

    print(f"PPL chunks: {len(ppl_dataset)}, Completion: {len(completion_samples)}, Discharge: {len(discharge_samples)}")

    all_results = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Filter to only requested models
    requested = sys.argv[1:] if len(sys.argv) > 1 else None

    for name, path in MODELS:
        if requested and name not in requested:
            continue

        print(f"\n{'='*60}")
        print(f"  Evaluating: {name}")
        print(f"  Path: {path}")
        print(f"{'='*60}")

        try:
            model = load_model(path, device=device)
        except Exception as e:
            print(f"  [ERROR] Cannot load {name}: {e}")
            continue

        cfg = MIMICEvalConfig(
            model_path=path,
            max_samples=MAX_SAMPLES,
            output_dir=OUTPUT,
            max_new_tokens=128,
            temperature=0.0,
        )

        results = {"name": name, "path": path}

        # PPL
        print(f"\n--- PPL ---")
        loss, ppl = evaluate_clinical_ppl(model, ppl_loader, device)
        results["ppl"] = {"loss": loss, "ppl": ppl}
        print_ppl_results(name, loss, ppl)

        # Completion
        print(f"\n--- Completion ({len(completion_samples)} samples) ---")
        comp_metrics, comp_preds = evaluate_completion(model, tokenizer, completion_samples, cfg, device)
        results["completion"] = comp_metrics
        print_generation_results("completion", comp_metrics)

        # Discharge
        print(f"\n--- Discharge ({len(discharge_samples)} samples) ---")
        dis_metrics, dis_preds = evaluate_discharge(model, tokenizer, discharge_samples, cfg, device)
        results["discharge"] = dis_metrics
        print_generation_results("discharge", dis_metrics)

        all_results[name] = results

        # Save individual
        with open(os.path.join(OUTPUT, f"{name.replace('+','_').replace(' ','_')}.json"), "w") as f:
            json.dump(results, f, indent=2)

        # Free GPU
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Save combined
    with open(os.path.join(OUTPUT, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n\n{'='*100}")
    print(f"{'Model':<20} {'PPL':>8} {'C-R1':>8} {'C-R2':>8} {'C-RL':>8} {'D-R1':>8} {'D-R2':>8} {'D-RL':>8}")
    print(f"{'='*100}")
    for name, r in all_results.items():
        ppl_val = r.get("ppl", {}).get("ppl", 0)
        cr1 = r.get("completion", {}).get("rouge1", 0) * 100
        cr2 = r.get("completion", {}).get("rouge2", 0) * 100
        crl = r.get("completion", {}).get("rougeL", 0) * 100
        dr1 = r.get("discharge", {}).get("rouge1", 0) * 100
        dr2 = r.get("discharge", {}).get("rouge2", 0) * 100
        drl = r.get("discharge", {}).get("rougeL", 0) * 100
        print(f"{name:<20} {ppl_val:>8.2f} {cr1:>8.2f} {cr2:>8.2f} {crl:>8.2f} {dr1:>8.2f} {dr2:>8.2f} {drl:>8.2f}")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
