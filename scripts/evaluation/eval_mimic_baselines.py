#!/usr/bin/env python3
"""
Evaluate external biomedical baseline models on MIMIC-IV clinical tasks:
  - Note Completion (ROUGE)
  - Discharge Summary Generation (ROUGE)

Each model runs in a subprocess to prevent CUDA error cascading.
"""
# --- Project root resolution (auto-generated) ---
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')
_sys.path.insert(0, _PROJECT_ROOT)
_os.chdir(_PROJECT_ROOT)
# --- End project root resolution ---

import json, os, sys, subprocess

EXTERNAL_MODELS = {
    "BioGPT": "microsoft/biogpt",
    "BioGPT-Large": "microsoft/BioGPT-Large",
    "BioGPT-Large-PubMedQA": "microsoft/BioGPT-Large-PubMedQA",
    "BioMedLM": "stanford-crfm/BioMedLM",
    "Bio-Medical-Llama-3.2-1B": "ContactDoctor/Bio-Medical-Llama-3-2-1B-CoT-012025",
    "Meditron3-Gemma2-2B": "OpenMeditron/Meditron3-Gemma2-2B",
    "Gemma3-finetune": "kunjcr2/gemma3_finetune",
}

RESULTS_DIR = os.path.join(_PROJECT_ROOT, "evaluation_results/mimic_baselines")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load existing results to skip already-evaluated models
    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    all_results = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            all_results = json.load(f)

    for model_name, model_id in EXTERNAL_MODELS.items():
        # Skip if already successfully evaluated
        if model_name in all_results and "error" not in all_results[model_name]:
            print(f"[SKIP] {model_name} already evaluated")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name} ({model_id})")
        print(f"{'='*60}")

        out_path = os.path.join(RESULTS_DIR, f"{model_name}.json")

        # Larger models get smaller batch to avoid OOM
        bs = 8 if "2B" in model_name or "2.7" in model_name else 16

        # Run in subprocess for CUDA isolation
        result = subprocess.run(
            [sys.executable, os.path.join(_PROJECT_ROOT, "scripts/evaluation/eval_mimic_single_baseline.py"),
             "--model_name", model_name,
             "--model_id", model_id,
             "--output", out_path,
             "--batch_size", str(bs)],
            capture_output=False,
            timeout=7200,  # 2 hour timeout per model
        )

        if result.returncode == 0 and os.path.exists(out_path):
            with open(out_path) as f:
                all_results[model_name] = json.load(f)
            print(f"  [OK] {model_name}")
        else:
            all_results[model_name] = {"error": f"subprocess exit code {result.returncode}"}
            print(f"  [FAIL] {model_name}")

        # Save summary after each model
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n{'='*90}")
    print("SUMMARY: External Baselines on MIMIC-IV (500 samples)")
    print(f"{'='*90}")
    print(f"{'Model':<30} {'C-R1':>6} {'C-R2':>6} {'C-RL':>6} {'D-R1':>6} {'D-R2':>6} {'D-RL':>6}")
    print("-" * 90)
    for name, res in all_results.items():
        if "error" in res:
            print(f"{name:<30} ERROR: {res['error'][:50]}")
            continue
        c = res["completion"]
        d = res["discharge"]
        print(f"{name:<30} {c['rouge1']*100:>6.2f} {c['rouge2']*100:>6.2f} {c['rougeL']*100:>6.2f} "
              f"{d['rouge1']*100:>6.2f} {d['rouge2']*100:>6.2f} {d['rougeL']*100:>6.2f}")


if __name__ == "__main__":
    main()
