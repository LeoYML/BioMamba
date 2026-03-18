#!/usr/bin/env python3
"""
Evaluate ALL ablation models (mixing ratios) on 3 domains: PubMed, Wiki, C4.
Includes: base, pure PubMed, C4-only, Wiki-only, and all C4+Wiki combinations.
Saves results incrementally to JSON.
"""
# --- Project root resolution (auto-generated) ---
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')
_sys.path.insert(0, _PROJECT_ROOT)
_os.chdir(_PROJECT_ROOT)
# --- End project root resolution ---

import json, math, os, sys, torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk, load_dataset
from tqdm import tqdm

sys.path.insert(0, "/data/BioMamba")
from ft_biomamba.model import load_model, load_tokenizer

DEVICE = "cuda:0"
BATCH = 32
SEQ_LEN = 1024
N_SEQS = 1000

RESULTS_DIR = os.path.join(_PROJECT_ROOT, "evaluation_results")
RESULTS_JSON = os.path.join(RESULTS_DIR, "ablation_mixing_ratios.json")

# All ablation models: (name, c4%, wiki%, checkpoint_path)
MODELS = [
    # Base (no CPT)
    ("Base (no CPT)",            0, 0, "state-spaces/mamba2-130m"),
    # Pure PubMed (CPT, no mixing)
    ("PubMed-only",              0, 0, os.path.join(_PROJECT_ROOT, "checkpoints/pubmed_only/biomamba_cpt_singledoc_mamba2-130m/best_model")),
    # C4-only
    ("C4-only 5%",               5, 0, os.path.join(_PROJECT_ROOT, "checkpoints/mix_c4only5/biomamba_cpt_singledoc_mamba2-130m/best_model")),
    ("C4-only 10%",             10, 0, os.path.join(_PROJECT_ROOT, "checkpoints/mixed/biomamba_cpt_singledoc_mamba2-130m/best_model")),
    ("C4-only 20%",             20, 0, os.path.join(_PROJECT_ROOT, "checkpoints/mix_c4only20/biomamba_cpt_singledoc_mamba2-130m/best_model")),
    ("C4-only 30%",             30, 0, os.path.join(_PROJECT_ROOT, "checkpoints/mix_c4only30/biomamba_cpt_singledoc_mamba2-130m/best_model")),
    # Wiki-only
    ("Wiki-only 10%",            0,10, os.path.join(_PROJECT_ROOT, "checkpoints/ablation_wiki10/biomamba_cpt_singledoc_mamba2-130m/best_model")),
    ("Wiki-only 20%",            0,20, os.path.join(_PROJECT_ROOT, "checkpoints/ablation_wiki20/biomamba_cpt_singledoc_mamba2-130m/best_model")),
    ("Wiki-only 30%",            0,30, os.path.join(_PROJECT_ROOT, "checkpoints/ablation_wiki30/biomamba_cpt_singledoc_mamba2-130m/best_model")),
    # C4 + Wiki mixed
    ("5%C4 + 5%Wiki",            5, 5, os.path.join(_PROJECT_ROOT, "checkpoints/mix_c45_wiki5/biomamba_cpt_singledoc_mamba2-130m/best_model")),
    ("5%C4 + 10%Wiki",           5,10, os.path.join(_PROJECT_ROOT, "checkpoints/mix_c45_wiki10/biomamba_cpt_singledoc_mamba2-130m/best_model")),
    ("10%C4 + 5%Wiki",          10, 5, os.path.join(_PROJECT_ROOT, "checkpoints/mix_c410_wiki5/biomamba_cpt_singledoc_mamba2-130m/best_model")),
    ("10%C4 + 10%Wiki",         10,10, os.path.join(_PROJECT_ROOT, "checkpoints/mixed_wiki/biomamba_cpt_singledoc_mamba2-130m/best_model")),
    ("10%C4 + 20%Wiki",         10,20, os.path.join(_PROJECT_ROOT, "checkpoints/mix_c410_wiki20/biomamba_cpt_singledoc_mamba2-130m/best_model")),
    ("15%C4 + 15%Wiki",         15,15, os.path.join(_PROJECT_ROOT, "checkpoints/mix_c415_wiki15/biomamba_cpt_singledoc_mamba2-130m/best_model")),
    ("20%C4 + 10%Wiki",         20,10, os.path.join(_PROJECT_ROOT, "checkpoints/mix_c420_wiki10/biomamba_cpt_singledoc_mamba2-130m/best_model")),
    ("20%C4 + 20%Wiki",         20,20, os.path.join(_PROJECT_ROOT, "checkpoints/mix_c420_wiki20/biomamba_cpt_singledoc_mamba2-130m/best_model")),
]


class TokenizedTextDataset(Dataset):
    def __init__(self, input_ids_list):
        self.data = input_ids_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return {"input_ids": torch.tensor(self.data[i], dtype=torch.long)}


def build_wiki_dataset(tokenizer, n_seqs=N_SEQS, seq_len=SEQ_LEN):
    print("Loading wikitext-103 test split...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test", trust_remote_code=True)
    texts = [ex["text"] for ex in ds if len(ex["text"].strip()) > 100]
    all_ids = []
    for text in texts:
        all_ids.extend(tokenizer.encode(text))
    chunks = [all_ids[i:i+seq_len] for i in range(0, len(all_ids)-seq_len, seq_len)]
    chunks = chunks[:n_seqs]
    print(f"  wikitext: {len(chunks)} chunks of {seq_len} tokens")
    return TokenizedTextDataset(chunks)


def build_c4_dataset(tokenizer, n_seqs=N_SEQS, seq_len=SEQ_LEN):
    print("Loading C4 validation split (streaming)...")
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True, trust_remote_code=True)
    all_ids = []
    for ex in ds:
        all_ids.extend(tokenizer.encode(ex["text"]))
        if len(all_ids) >= n_seqs * seq_len * 2:
            break
    chunks = [all_ids[i:i+seq_len] for i in range(0, len(all_ids)-seq_len, seq_len)]
    chunks = chunks[:n_seqs]
    print(f"  C4: {len(chunks)} chunks of {seq_len} tokens")
    return TokenizedTextDataset(chunks)


def build_pubmed_dataset():
    print("Loading PubMed test split...")
    ds = load_from_disk(os.path.join(_PROJECT_ROOT, "data/pubmed_medline_tokenized_1024/test"))
    print(f"  PubMed: {len(ds):,} sequences")
    return ds, True


def evaluate(model, loader, device, has_attn_mask=False):
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="  eval", leave=False):
            inputs = batch["input_ids"].to(device)
            labels = inputs.clone()
            if has_attn_mask and "attention_mask" in batch:
                mask = batch["attention_mask"].to(device)
                labels[mask == 0] = -100
            with autocast("cuda", dtype=torch.bfloat16):
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


def collate_fn(batch):
    return {"input_ids": torch.stack([b["input_ids"] for b in batch])}


def _save(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)


def main():
    tokenizer = load_tokenizer()

    # Build datasets once
    pubmed_ds, pubmed_attn = build_pubmed_dataset()
    wiki_ds = build_wiki_dataset(tokenizer)
    c4_ds = build_c4_dataset(tokenizer)

    pubmed_loader = DataLoader(pubmed_ds, batch_size=BATCH, shuffle=False,
                                num_workers=4, pin_memory=True)
    wiki_loader = DataLoader(wiki_ds, batch_size=BATCH, shuffle=False,
                              num_workers=2, pin_memory=True, collate_fn=collate_fn)
    c4_loader = DataLoader(c4_ds, batch_size=BATCH, shuffle=False,
                            num_workers=2, pin_memory=True, collate_fn=collate_fn)

    # Load existing results to support incremental evaluation
    results = {}
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results from {RESULTS_JSON}")

    for name, c4_pct, wiki_pct, path in MODELS:
        # Skip if already evaluated
        if name in results and results[name].get("pubmed_ppl") is not None:
            print(f"\nSKIP {name}: already evaluated")
            continue

        if not os.path.exists(path) and not path.startswith("state-spaces/"):
            print(f"\nSKIP {name}: {path} not found")
            results[name] = {"c4_pct": c4_pct, "wiki_pct": wiki_pct,
                             "pubmed_ppl": None, "wiki_ppl": None, "c4_ppl": None,
                             "status": "checkpoint_not_found"}
            _save(results)
            continue

        print(f"\n{'='*60}")
        print(f"  {name} (C4={c4_pct}%, Wiki={wiki_pct}%)")
        print(f"  {path}")
        print(f"{'='*60}")

        try:
            model = load_model(path, device=DEVICE)

            print("  [PubMed]")
            pl, pp = evaluate(model, pubmed_loader, DEVICE, has_attn_mask=True)
            print(f"    loss={pl:.4f}  ppl={pp:.2f}")

            print("  [Wikipedia]")
            wl, wp = evaluate(model, wiki_loader, DEVICE)
            print(f"    loss={wl:.4f}  ppl={wp:.2f}")

            print("  [C4]")
            cl, cp = evaluate(model, c4_loader, DEVICE)
            print(f"    loss={cl:.4f}  ppl={cp:.2f}")

            results[name] = {
                "c4_pct": c4_pct, "wiki_pct": wiki_pct,
                "pubmed_loss": pl, "pubmed_ppl": pp,
                "wiki_loss": wl, "wiki_ppl": wp,
                "c4_loss": cl, "c4_ppl": cp,
                "status": "ok"
            }
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = {"c4_pct": c4_pct, "wiki_pct": wiki_pct,
                             "pubmed_ppl": None, "wiki_ppl": None, "c4_ppl": None,
                             "status": f"error: {e}"}

        _save(results)

    # ── Summary table ──
    print(f"\n{'='*90}")
    print(f"{'Model':<22} {'C4%':>4} {'Wiki%':>5} {'PubMed PPL':>12} {'Wiki PPL':>10} {'C4 PPL':>10}")
    print("-"*90)
    for name, c4_pct, wiki_pct, _ in MODELS:
        r = results.get(name, {})
        pp = r.get("pubmed_ppl")
        wp = r.get("wiki_ppl")
        cp = r.get("c4_ppl")
        def fmt(v):
            return f"{v:.2f}" if v else "N/A"
        print(f"{name:<22} {c4_pct:>4} {wiki_pct:>5} {fmt(pp):>12} {fmt(wp):>10} {fmt(cp):>10}")
    print("="*90)

    _save(results)
    print(f"\nResults saved to {RESULTS_JSON}")


if __name__ == "__main__":
    main()
