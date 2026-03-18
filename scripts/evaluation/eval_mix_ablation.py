#!/usr/bin/env python3
"""
Evaluate PubMed PPL for all mix ablation models.
Compares against baseline (10%C4+10%Wiki) and wiki-only ablations.
"""
# --- Project root resolution (auto-generated) ---
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')
_sys.path.insert(0, _PROJECT_ROOT)
_os.chdir(_PROJECT_ROOT)
# --- End project root resolution ---

import math, sys, torch, os
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tqdm import tqdm

sys.path.insert(0, "/data/BioMamba")
from ft_biomamba.model import load_model

BATCH  = 32
DEVICE = "cuda:0"

MODELS = [
    # Baselines
    ("130m-base",        "state-spaces/mamba2-130m"),
    ("10%C4+10%Wiki",    os.path.join(_PROJECT_ROOT, "checkpoints/mixed_wiki/biomamba_cpt_singledoc_mamba2-130m/best_model")),
    # Previous wiki-only ablations
    ("Wiki-only-10%",    os.path.join(_PROJECT_ROOT, "checkpoints/ablation_wiki10/biomamba_cpt_singledoc_mamba2-130m/best_model")),
    ("Wiki-only-20%",    os.path.join(_PROJECT_ROOT, "checkpoints/ablation_wiki20/biomamba_cpt_singledoc_mamba2-130m/best_model")),
    ("Wiki-only-30%",    os.path.join(_PROJECT_ROOT, "checkpoints/ablation_wiki30/biomamba_cpt_singledoc_mamba2-130m/best_model")),
    # New C4-only ablations
    ("C4-only-5%",       os.path.join(_PROJECT_ROOT, "checkpoints/mix_c4only5/biomamba_cpt_singledoc_mamba2-130m/best_model")),
    ("C4-only-20%",      os.path.join(_PROJECT_ROOT, "checkpoints/mix_c4only20/biomamba_cpt_singledoc_mamba2-130m/best_model")),
    ("C4-only-30%",      os.path.join(_PROJECT_ROOT, "checkpoints/mix_c4only30/biomamba_cpt_singledoc_mamba2-130m/best_model")),
    # New mixed ablations
    ("5%C4+5%Wiki",      os.path.join(_PROJECT_ROOT, "checkpoints/mix_c45_wiki5/biomamba_cpt_singledoc_mamba2-130m/best_model")),
    ("20%C4+10%Wiki",    os.path.join(_PROJECT_ROOT, "checkpoints/mix_c420_wiki10/biomamba_cpt_singledoc_mamba2-130m/best_model")),
]


def evaluate(model, loader, device):
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            inputs = batch["input_ids"].to(device)
            labels = inputs.clone()
            if "attention_mask" in batch:
                mask = batch["attention_mask"].to(device)
                labels[mask == 0] = -100
            with autocast(dtype=torch.bfloat16):
                out = model(inputs)
                loss = F.cross_entropy(
                    out.logits[:, :-1, :].reshape(-1, out.logits.size(-1)),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                )
            if torch.isfinite(loss):
                total_loss += loss.item()
                n += 1
    avg = total_loss / n if n > 0 else float("inf")
    return avg, math.exp(avg)


def main():
    test_ds = load_from_disk(os.path.join(_PROJECT_ROOT, "data/pubmed_medline_tokenized_1024/test"))
    test_ds.set_format("torch")
    loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False,
                        num_workers=4, pin_memory=True)
    print(f"PubMed test set: {len(test_ds):,} sequences\n")

    results = []
    for name, path in MODELS:
        if not os.path.exists(path) and not path.startswith("state-spaces/"):
            print(f"  SKIP {name}: {path} not found")
            results.append((name, None, None))
            continue
        print(f"\n{'='*60}")
        print(f"  {name}  —  {path}")
        print(f"{'='*60}")
        try:
            model = load_model(path, device=DEVICE)
            loss, ppl = evaluate(model, loader, DEVICE)
            print(f"  loss={loss:.4f}  ppl={ppl:.2f}")
            results.append((name, loss, ppl))
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((name, None, None))

    # Find baseline
    baseline_ppl = None
    for name, loss, ppl in results:
        if name == "10%C4+10%Wiki" and ppl:
            baseline_ppl = ppl
            break

    # Full results table
    print("\n" + "=" * 65)
    print(f"{'Model':<20} {'Loss':>8} {'PubMed PPL':>12} {'vs Baseline':>12}")
    print("-" * 65)
    for name, loss, ppl in results:
        if loss is not None:
            if name == "10%C4+10%Wiki":
                tag = "(baseline)"
            elif baseline_ppl:
                delta_pct = (ppl - baseline_ppl) / baseline_ppl * 100
                tag = f"{delta_pct:+.1f}%"
            else:
                tag = ""
            print(f"{name:<20} {loss:>8.4f} {ppl:>12.2f} {tag:>12}")
        else:
            print(f"{name:<20} {'N/A':>8} {'N/A':>12} {'SKIP':>12}")
    print("=" * 65)

    # Ranking
    valid = [(n, l, p) for n, l, p in results if p is not None and n != "130m-base"]
    valid.sort(key=lambda x: x[2])
    print(f"\nRanking (best → worst PubMed PPL):")
    for i, (name, loss, ppl) in enumerate(valid, 1):
        marker = " ★" if ppl == valid[0][2] else ""
        print(f"  {i}. {name:<20} PPL={ppl:.2f}{marker}")


if __name__ == "__main__":
    main()
