"""
Evaluate base vs CPT model PPL on three domains:
  - PubMed (in-domain)
  - Wikipedia (wikitext-103 test)
  - General (C4 validation)

Shows whether CPT improves bio domain while preserving general language ability.
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

RESULTS_DIR = os.path.join(_PROJECT_ROOT, "evaluation_results")
RESULTS_JSON = os.path.join(RESULTS_DIR, "ppl_domain_comparison.json")


def _save(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [saved → {RESULTS_JSON}]")

sys.path.insert(0, "/data/BioMamba")
from ft_biomamba.model import load_model, load_tokenizer

DEVICE   = "cuda:0"
BATCH    = 16
SEQ_LEN  = 1024
N_SEQS   = 1000   # sequences per domain for wiki/c4

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


# ── Dataset helpers ──────────────────────────────────────────────────────────

class TokenizedTextDataset(Dataset):
    def __init__(self, input_ids_list):
        self.data = input_ids_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return {"input_ids": torch.tensor(self.data[i], dtype=torch.long)}


def build_wiki_dataset(tokenizer, n_seqs=N_SEQS, seq_len=SEQ_LEN):
    """wikitext-103-raw-v1 test split → tokenized chunks."""
    print("Loading wikitext-103 test split...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test", trust_remote_code=True)
    texts = [ex["text"] for ex in ds if len(ex["text"].strip()) > 100]

    all_ids = []
    for text in texts:
        ids = tokenizer.encode(text)
        all_ids.extend(ids)

    chunks = [all_ids[i:i+seq_len] for i in range(0, len(all_ids)-seq_len, seq_len)]
    chunks = chunks[:n_seqs]
    print(f"  wikitext: {len(chunks)} chunks of {seq_len} tokens")
    return TokenizedTextDataset(chunks)


def build_c4_dataset(tokenizer, n_seqs=N_SEQS, seq_len=SEQ_LEN):
    """C4 validation split → tokenized chunks."""
    print("Loading C4 validation split (streaming)...")
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True,
                      trust_remote_code=True)

    all_ids = []
    for ex in ds:
        ids = tokenizer.encode(ex["text"])
        all_ids.extend(ids)
        if len(all_ids) >= n_seqs * seq_len * 2:
            break

    chunks = [all_ids[i:i+seq_len] for i in range(0, len(all_ids)-seq_len, seq_len)]
    chunks = chunks[:n_seqs]
    print(f"  C4: {len(chunks)} chunks of {seq_len} tokens")
    return TokenizedTextDataset(chunks)


def build_pubmed_dataset():
    """Use the existing pre-tokenized PubMed test split."""
    print("Loading PubMed test split...")
    ds = load_from_disk(os.path.join(_PROJECT_ROOT, "data/pubmed_mixed_10pct_general_10pct_wiki/test"))
    print(f"  PubMed: {len(ds):,} sequences")
    return ds, True  # flag: has attention_mask


# ── Evaluation ───────────────────────────────────────────────────────────────

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
    input_ids = torch.stack([b["input_ids"] for b in batch])
    return {"input_ids": input_ids}


def main():
    tokenizer = load_tokenizer()

    # Build datasets once
    pubmed_ds, pubmed_attn = build_pubmed_dataset()
    wiki_ds   = build_wiki_dataset(tokenizer)
    c4_ds     = build_c4_dataset(tokenizer)

    pubmed_loader = DataLoader(pubmed_ds, batch_size=BATCH, shuffle=False,
                               num_workers=4, pin_memory=True)
    wiki_loader   = DataLoader(wiki_ds,   batch_size=BATCH, shuffle=False,
                               num_workers=2, pin_memory=True, collate_fn=collate_fn)
    c4_loader     = DataLoader(c4_ds,     batch_size=BATCH, shuffle=False,
                               num_workers=2, pin_memory=True, collate_fn=collate_fn)

    results = {}  # {name: {pubmed, wiki, c4}}

    for name, path in MODELS:
        print(f"\n{'='*60}")
        print(f"  {name}  —  {path}")
        print(f"{'='*60}")
        try:
            model = load_model(path, device=DEVICE)
            print("  [PubMed]")
            pl, pp = evaluate(model, pubmed_loader, DEVICE, has_attn_mask=True)
            print(f"    loss={pl:.4f}  ppl={pp:.4f}")
            print("  [Wikipedia]")
            wl, wp = evaluate(model, wiki_loader, DEVICE)
            print(f"    loss={wl:.4f}  ppl={wp:.4f}")
            print("  [C4/General]")
            cl, cp = evaluate(model, c4_loader, DEVICE)
            print(f"    loss={cl:.4f}  ppl={cp:.4f}")
            results[name] = {"pubmed": pp, "wiki": wp, "c4": cp,
                             "pubmed_loss": pl, "wiki_loss": wl, "c4_loss": cl}
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = {"pubmed": None, "wiki": None, "c4": None}
        _save(results)  # incremental save after each model

    # ── Summary table ──
    print("\n" + "="*70)
    print(f"{'Model':<15} {'PubMed PPL':>12} {'Wiki PPL':>10} {'C4 PPL':>10}")
    print("-"*70)
    for name, r in results.items():
        def fmt(v): return f"{v:10.4f}" if v else "     ERROR"
        print(f"{name:<15} {fmt(r['pubmed'])} {fmt(r['wiki'])} {fmt(r['c4'])}")

    # ── Delta table (base → CPT) ──
    print("\n" + "="*80)
    print(f"{'Size':<8} {'Domain':<10} {'Base PPL':>10} {'CPT PPL':>10} {'↓ PPL':>8} {'Improv%':>8}")
    print("-"*80)
    pairs = [("130m","130m-base","130m-CPT"),
             ("370m","370m-base","370m-CPT"),
             ("780m","780m-base","780m-CPT"),
             ("1.3b","1.3b-base","1.3b-CPT"),
             ("2.7b","2.7b-base","2.7b-CPT")]
    for sz, base_name, cpt_name in pairs:
        if base_name not in results or cpt_name not in results:
            continue
        for domain in ["pubmed", "wiki", "c4"]:
            bp = results[base_name][domain]
            cp = results[cpt_name][domain]
            if bp and cp:
                delta = bp - cp
                pct   = delta / bp * 100
                sign  = "+" if pct < 0 else ""
                print(f"{sz:<8} {domain:<10} {bp:>10.4f} {cp:>10.4f} {delta:>8.4f} {sign}{pct:>6.1f}%")
    print("="*80)

    _save(results)
    # also write a human-readable TSV
    tsv_path = os.path.join(RESULTS_DIR, "ppl_domain_comparison.tsv")
    with open(tsv_path, "w") as f:
        f.write("model\tpubmed_ppl\twiki_ppl\tc4_ppl\n")
        for name, r in results.items():
            def fv(v): return f"{v:.4f}" if v else "ERROR"
            f.write(f"{name}\t{fv(r['pubmed'])}\t{fv(r['wiki'])}\t{fv(r['c4'])}\n")
    print(f"\nSaved to {RESULTS_JSON}")
    print(f"Saved to {tsv_path}")


if __name__ == "__main__":
    main()
