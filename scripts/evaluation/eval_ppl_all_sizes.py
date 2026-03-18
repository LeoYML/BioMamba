"""
Compare base vs CPT model perplexity on the test set for all sizes.
Uses the same eval logic as the trainer (full test set).
"""
# --- Project root resolution (auto-generated) ---
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')
_sys.path.insert(0, _PROJECT_ROOT)
_os.chdir(_PROJECT_ROOT)
# --- End project root resolution ---

import math, os, sys, torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tqdm import tqdm

sys.path.insert(0, "/data/BioMamba")
from ft_biomamba.model import load_model, load_tokenizer

DATA_DIR = os.path.join(_PROJECT_ROOT, "data/pubmed_mixed_10pct_general_10pct_wiki")
BATCH    = 32
DEVICE   = "cuda:0"

MODELS = [
    ("130m-base", "state-spaces/mamba2-130m"),
    ("130m-CPT",  os.path.join(_PROJECT_ROOT, "checkpoints/mixed_wiki/biomamba_cpt_singledoc_mamba2-130m/best_model")),
    ("370m-base", "state-spaces/mamba2-370m"),
    ("370m-CPT",  os.path.join(_PROJECT_ROOT, "checkpoints/370m_mixed_wiki/biomamba_cpt_singledoc_mamba2-370m/best_model")),
    ("780m-base", "state-spaces/mamba2-780m"),
    ("780m-CPT",  os.path.join(_PROJECT_ROOT, "checkpoints/780m_mixed_wiki/biomamba_cpt_singledoc_mamba2-780m/best_model")),
    ("1.3b-base", "state-spaces/mamba2-1.3b"),
    ("1.3b-CPT",  os.path.join(_PROJECT_ROOT, "checkpoints/1.3b_mixed_wiki/biomamba_cpt_singledoc_mamba2-1.3b/best_model")),
    ("2.7b-base", "state-spaces/mamba2-2.7b"),
    ("2.7b-CPT",  os.path.join(_PROJECT_ROOT, "checkpoints/2.7b_mixed_wiki/biomamba_cpt_singledoc_mamba2-2.7b/best_model")),
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
    test_ds = load_from_disk(f"{DATA_DIR}/test")
    loader  = DataLoader(test_ds, batch_size=BATCH, shuffle=False,
                         num_workers=4, pin_memory=True)
    print(f"Test set: {len(test_ds):,} sequences\n")

    results = []
    for name, path in MODELS:
        print(f"\n{'='*60}")
        print(f"  {name}  —  {path}")
        print(f"{'='*60}")
        try:
            model = load_model(path, device=DEVICE)
            loss, ppl = evaluate(model, loader, DEVICE)
            print(f"  loss={loss:.4f}  ppl={ppl:.4f}")
            results.append((name, loss, ppl))
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((name, None, None))

    # Full results table
    print("\n" + "="*45)
    print(f"{'Model':<15} {'Loss':>8} {'PPL':>10}")
    print("-"*45)
    for name, loss, ppl in results:
        if loss is not None:
            print(f"{name:<15} {loss:>8.4f} {ppl:>10.4f}")
        else:
            print(f"{name:<15} {'ERROR':>8} {'ERROR':>10}")

    # Delta table
    print("\n" + "="*58)
    print(f"{'Size':<8} {'Base PPL':>10} {'CPT PPL':>10} {'↓ PPL':>8} {'Improv%':>8}")
    print("-"*58)
    pairs = [("130m",0,1),("370m",2,3),("780m",4,5),("1.3b",6,7),("2.7b",8,9)]
    for sz, bi, ci in pairs:
        bp = results[bi][2]
        cp = results[ci][2]
        if bp and cp:
            delta = bp - cp
            pct   = delta / bp * 100
            print(f"{sz:<8} {bp:>10.4f} {cp:>10.4f} {delta:>8.4f} {pct:>7.1f}%")
    print("="*58)

if __name__ == "__main__":
    main()
