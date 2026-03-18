#!/usr/bin/env python3
"""
Evaluate PubMed PPL for wiki-only ablation models vs baseline (10%C4+10%Wiki).
Tests on pure PubMed test set to measure domain-specific quality.
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
from ft_biomamba.model import load_model, load_tokenizer

BATCH  = 32
DEVICE = "cuda:0"

MODELS = [
    ("130m-base",       "state-spaces/mamba2-130m"),
    ("10%C4+10%Wiki",   os.path.join(_PROJECT_ROOT, "checkpoints/mixed_wiki/biomamba_cpt_singledoc_mamba2-130m/best_model")),
    ("Wiki-only-10%",   os.path.join(_PROJECT_ROOT, "checkpoints/ablation_wiki10/biomamba_cpt_singledoc_mamba2-130m/best_model")),
    ("Wiki-only-20%",   os.path.join(_PROJECT_ROOT, "checkpoints/ablation_wiki20/biomamba_cpt_singledoc_mamba2-130m/best_model")),
    ("Wiki-only-30%",   os.path.join(_PROJECT_ROOT, "checkpoints/ablation_wiki30/biomamba_cpt_singledoc_mamba2-130m/best_model")),
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
    # Use pure PubMed test set for fair comparison
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

    # Results table
    print("\n" + "=" * 60)
    print(f"{'Model':<20} {'Loss':>8} {'PubMed PPL':>12}")
    print("-" * 60)
    baseline_ppl = None
    for name, loss, ppl in results:
        if loss is not None:
            marker = ""
            if name == "10%C4+10%Wiki":
                baseline_ppl = ppl
                marker = " (baseline)"
            elif baseline_ppl and ppl < baseline_ppl:
                marker = " ✓ better"
            elif baseline_ppl and ppl > baseline_ppl:
                marker = " ✗ worse"
            print(f"{name:<20} {loss:>8.4f} {ppl:>12.2f}{marker}")
        else:
            print(f"{name:<20} {'N/A':>8} {'N/A':>12}")
    print("=" * 60)

    # Delta table vs baseline
    if baseline_ppl:
        print(f"\nComparison vs baseline (10%C4+10%Wiki, PPL={baseline_ppl:.2f}):")
        print(f"{'Model':<20} {'PPL':>10} {'Δ PPL':>10} {'Change':>10}")
        print("-" * 55)
        for name, loss, ppl in results:
            if loss is not None and name != "130m-base" and name != "10%C4+10%Wiki":
                delta = ppl - baseline_ppl
                pct = delta / baseline_ppl * 100
                sign = "+" if delta > 0 else ""
                print(f"{name:<20} {ppl:>10.2f} {sign}{delta:>9.2f} {sign}{pct:>8.1f}%")
        print("=" * 55)


if __name__ == "__main__":
    main()
