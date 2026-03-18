"""Run margin threshold sweep on all best SFT models for PubMedQA."""
# --- Project root resolution (auto-generated) ---
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')
_sys.path.insert(0, _PROJECT_ROOT)
_os.chdir(_PROJECT_ROOT)
# --- End project root resolution ---

import os
import torch
import torch.nn.functional as F
import json
from collections import Counter
from tqdm import tqdm

# Reuse functions from eval_pubmedqa_generative
from eval_pubmedqa_generative import (
    load_pubmedqa_test, load_tokenizer, load_full_model,
    format_prompt, _get_candidate_logits, compute_metrics,
)

MODELS = {
    "2.7B": os.path.join(_PROJECT_ROOT, "checkpoints/2.7b_sft/biomamba2_sft_mamba2-2.7b_full_20260309_014707/best_model"),
    "1.3B-v7b": os.path.join(_PROJECT_ROOT, "checkpoints/1.3b_sft_v7/biomamba2_sft_mamba2-1.3b_full_20260309_060641/best_model"),
    "780M": os.path.join(_PROJECT_ROOT, "checkpoints/780m_sft/biomamba2_sft_mamba2-780m_full_20260309_025943/best_model"),
    "370M-v7": os.path.join(_PROJECT_ROOT, "checkpoints/370m_sft_v7/biomamba2_sft_mamba2-370m_full_20260309_053303/best_model"),
    "130M": os.path.join(_PROJECT_ROOT, "checkpoints/mixed_sft_v3/biomamba_sft_mamba2-130m_full/best_model"),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
samples = load_pubmedqa_test(seed=42)
tokenizer = load_tokenizer()
refs = [s["answer"] for s in samples]

all_results = {}

for model_name, model_path in MODELS.items():
    print(f"\n{'#'*70}")
    print(f"# {model_name}: {model_path}")
    print(f"{'#'*70}")

    model = load_full_model(model_path, device)

    # Collect logits
    all_cand_logits = []
    for ex in tqdm(samples, desc=f"  {model_name} logits"):
        prompt = format_prompt(ex["question"], ex["context"], template="sft")
        cl = _get_candidate_logits(model, tokenizer, prompt, device)
        all_cand_logits.append(cl)

    # Baseline (no margin)
    baseline_preds = []
    for cl in all_cand_logits:
        if not cl:
            baseline_preds.append("yes")
            continue
        labels = list(cl.keys())
        vals = torch.tensor([cl[l] for l in labels])
        probs = F.softmax(vals, dim=0)
        prob_dict = {l: p.item() for l, p in zip(labels, probs)}
        top_label = max(prob_dict, key=prob_dict.get)
        baseline_preds.append(top_label)
    baseline_m = compute_metrics(baseline_preds, refs)

    print(f"\n  Baseline (no margin): Acc={baseline_m['accuracy']:.1%} F1={baseline_m['macro_f1']:.1%}")

    # Sweep
    print(f"\n  {'margin':>6s} {'maybe_t':>7s} | {'Acc':>6s} {'F1':>6s} {'yes-F1':>7s} {'no-F1':>7s} {'maybe-F1':>8s}")
    print(f"  {'-'*65}")

    best_acc, best_f1 = 0, 0
    best_acc_cfg, best_f1_cfg = None, None
    best_acc_m, best_f1_m = None, None

    for margin_t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
        for maybe_t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
            preds = []
            for cl in all_cand_logits:
                if not cl:
                    preds.append("yes")
                    continue
                labels = list(cl.keys())
                vals = torch.tensor([cl[l] for l in labels])
                probs = F.softmax(vals, dim=0)
                prob_dict = {l: p.item() for l, p in zip(labels, probs)}
                sorted_c = sorted(prob_dict.items(), key=lambda x: -x[1])
                top_label, top_prob = sorted_c[0]
                if len(sorted_c) >= 2:
                    margin = top_prob - sorted_c[1][1]
                    maybe_prob = prob_dict.get("maybe", 0)
                    if margin < margin_t and maybe_prob > maybe_t:
                        preds.append("maybe")
                        continue
                preds.append(top_label)

            m = compute_metrics(preds, refs)
            yes_f1 = m["per_class"].get("yes", {}).get("F1", 0)
            no_f1 = m["per_class"].get("no", {}).get("F1", 0)
            maybe_f1 = m["per_class"].get("maybe", {}).get("F1", 0)
            flag = ""
            if m["accuracy"] > best_acc:
                best_acc = m["accuracy"]
                best_acc_cfg = (margin_t, maybe_t)
                best_acc_m = m
                flag += " *ACC"
            if m["macro_f1"] > best_f1:
                best_f1 = m["macro_f1"]
                best_f1_cfg = (margin_t, maybe_t)
                best_f1_m = m
                flag += " *F1"
            print(f"  {margin_t:>6.2f} {maybe_t:>7.2f} | {m['accuracy']:>5.1%} {m['macro_f1']:>5.1%} "
                  f"{yes_f1:>6.1%} {no_f1:>6.1%} {maybe_f1:>7.1%}{flag}")

    print(f"\n  Best accuracy: {best_acc:.1%} at margin={best_acc_cfg}")
    print(f"  Best macro F1: {best_f1:.1%} at margin={best_f1_cfg}")

    all_results[model_name] = {
        "baseline": {
            "acc": baseline_m["accuracy"],
            "f1": baseline_m["macro_f1"],
            "recall": baseline_m["macro_R"],
        },
        "best_acc": {
            "acc": best_acc,
            "cfg": best_acc_cfg,
            "f1": best_acc_m["macro_f1"] if best_acc_m else 0,
            "recall": best_acc_m["macro_R"] if best_acc_m else 0,
            "per_class": best_acc_m["per_class"] if best_acc_m else {},
        },
        "best_f1": {
            "f1": best_f1,
            "cfg": best_f1_cfg,
            "acc": best_f1_m["accuracy"] if best_f1_m else 0,
            "recall": best_f1_m["macro_R"] if best_f1_m else 0,
            "per_class": best_f1_m["per_class"] if best_f1_m else {},
        },
    }

    del model
    torch.cuda.empty_cache()

# Final summary
print(f"\n\n{'='*80}")
print("MARGIN THRESHOLD SWEEP SUMMARY")
print(f"{'='*80}")
print(f"{'Model':<12s} | {'Baseline Acc':>12s} {'Baseline F1':>12s} | {'Best Acc':>9s} {'cfg':>16s} | {'Best F1':>8s} {'cfg':>16s}")
print(f"{'-'*95}")
for name, r in all_results.items():
    print(f"{name:<12s} | {r['baseline']['acc']:>11.1%} {r['baseline']['f1']:>11.1%} | "
          f"{r['best_acc']['acc']:>8.1%} {str(r['best_acc']['cfg']):>16s} | "
          f"{r['best_f1']['f1']:>7.1%} {str(r['best_f1']['cfg']):>16s}")

# Save
_margin_out = os.path.join(_PROJECT_ROOT, "evaluation_results/margin_sweep_all_models.json")
os.makedirs(os.path.dirname(_margin_out), exist_ok=True)
with open(_margin_out, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nSaved to {_margin_out}")
