"""
Evaluate external bio decoder-only models on three domains:
  - PubMed (in-domain)
  - Wikipedia (wikitext-103 test)
  - General (C4 validation)

Compares BioGPT, BioMedLM, Gemma3-finetune, etc. against BioMamba CPT.
Each model uses its own tokenizer for fair comparison.
"""
# --- Project root resolution (auto-generated) ---
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')
_sys.path.insert(0, _PROJECT_ROOT)
_os.chdir(_PROJECT_ROOT)
# --- End project root resolution ---

import json, math, os, sys, torch, gc
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

RESULTS_DIR = os.path.join(_PROJECT_ROOT, "evaluation_results")
RESULTS_JSON = os.path.join(RESULTS_DIR, "ppl_external_models.json")

sys.path.insert(0, "/data/BioMamba")
from ft_biomamba.model import load_model as load_mamba_model, load_tokenizer as load_mamba_tokenizer

DEVICE = "cuda:0"
BATCH  = 8       # smaller batch for larger models
SEQ_LEN = 1024
N_SEQS  = 1000   # sequences per domain for wiki/c4
N_PUBMED = 1000  # PubMed sequences (from streaming)


def _save(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [saved -> {RESULTS_JSON}]")


# ── Raw text collection ──────────────────────────────────────────────────────

def collect_pubmed_texts(n=N_PUBMED):
    """Collect raw PubMed abstracts from streaming dataset."""
    print("Loading PubMed abstracts (streaming)...")
    ds = load_dataset("pubmed_qa", "pqa_artificial", split="train", streaming=True)
    texts = []
    for ex in ds:
        ctx = " ".join(ex.get("context", {}).get("contexts", []))
        if len(ctx) > 200:
            texts.append(ctx)
        if len(texts) >= n:
            break
    print(f"  PubMed: {len(texts)} abstracts collected")
    return texts


def collect_wiki_texts(n=N_SEQS):
    """Collect raw Wikipedia texts from wikitext-103."""
    print("Loading wikitext-103 test split...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    texts = [ex["text"] for ex in ds if len(ex["text"].strip()) > 200]
    # Concatenate short texts into longer chunks
    full_text = "\n".join(texts)
    # Split into ~2000 char chunks
    chunk_size = 2000
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
    chunks = [c for c in chunks if len(c) > 500][:n]
    print(f"  Wiki: {len(chunks)} text chunks")
    return chunks


def collect_c4_texts(n=N_SEQS):
    """Collect raw C4 texts."""
    print("Loading C4 validation split (streaming)...")
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    texts = []
    for ex in ds:
        if len(ex["text"]) > 200:
            texts.append(ex["text"][:3000])  # cap length
        if len(texts) >= n:
            break
    print(f"  C4: {len(texts)} texts")
    return texts


# ── Tokenized dataset ────────────────────────────────────────────────────────

class TokenizedChunkDataset(Dataset):
    """Tokenize raw texts into fixed-length chunks using a given tokenizer."""
    def __init__(self, texts, tokenizer, seq_len=SEQ_LEN, max_chunks=N_SEQS):
        all_ids = []
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            all_ids.extend(ids)
            if len(all_ids) >= max_chunks * seq_len * 2:
                break
        self.chunks = [
            torch.tensor(all_ids[i:i+seq_len], dtype=torch.long)
            for i in range(0, len(all_ids) - seq_len, seq_len)
        ][:max_chunks]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        return {"input_ids": self.chunks[i]}


def collate_fn(batch):
    return {"input_ids": torch.stack([b["input_ids"] for b in batch])}


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_causal(model, loader, device):
    """Standard causal LM perplexity evaluation."""
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="  eval", leave=False):
            inputs = batch["input_ids"].to(device)
            labels = inputs.clone()
            with autocast("cuda", dtype=torch.bfloat16):
                out = model(inputs)
                logits = out.logits if hasattr(out, 'logits') else out[0]
                loss = F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                )
            if torch.isfinite(loss):
                total_loss += loss.item()
                n += 1
    avg = total_loss / n if n > 0 else float("inf")
    return avg, math.exp(avg)


# ── Mamba evaluation (uses pre-tokenized data) ──────────────────────────────

def evaluate_mamba_on_texts(model_path, texts_dict, device=DEVICE):
    """Evaluate a Mamba model on raw texts using Mamba tokenizer."""
    tokenizer = load_mamba_tokenizer()
    model = load_mamba_model(model_path, device=device)

    results = {}
    for domain, texts in texts_dict.items():
        ds = TokenizedChunkDataset(texts, tokenizer)
        loader = DataLoader(ds, batch_size=BATCH, shuffle=False,
                           num_workers=2, pin_memory=True, collate_fn=collate_fn)
        print(f"  [{domain}] {len(ds)} chunks")
        loss, ppl = evaluate_causal(model, loader, device)
        print(f"    loss={loss:.4f}  ppl={ppl:.2f}")
        results[domain] = {"loss": loss, "ppl": ppl}

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return results


# ── HuggingFace model evaluation ────────────────────────────────────────────

def evaluate_hf_model(model_name, texts_dict, device=DEVICE, batch_size=BATCH):
    """Load a HuggingFace causal LM and evaluate on all domains."""
    print(f"  Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device,
    )
    model.eval()

    results = {}
    for domain, texts in texts_dict.items():
        ds = TokenizedChunkDataset(texts, tokenizer)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                           num_workers=2, pin_memory=True, collate_fn=collate_fn)
        print(f"  [{domain}] {len(ds)} chunks")
        loss, ppl = evaluate_causal(model, loader, device)
        print(f"    loss={loss:.4f}  ppl={ppl:.2f}")
        results[domain] = {"loss": loss, "ppl": ppl}

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    return results


# ── Model definitions ────────────────────────────────────────────────────────

MODELS = [
    # BioMamba CPT (best at each size for reference)
    ("BioMamba-130m-CPT", "mamba",
     os.path.join(_PROJECT_ROOT, "checkpoints/mixed_wiki/biomamba_cpt_singledoc_mamba2-130m/best_model")),
    ("BioMamba-2.7b-CPT", "mamba",
     os.path.join(_PROJECT_ROOT, "checkpoints/2.7b_mixed_wiki/biomamba_cpt_singledoc_mamba2-2.7b/best_model")),
    # External bio models
    ("BioGPT", "hf", "microsoft/biogpt"),
    ("BioGPT-Large", "hf", "microsoft/BioGPT-Large"),
    ("BioMedLM-2.7B", "hf", "stanford-crfm/BioMedLM"),
    ("Gemma3-finetune", "hf", "kunjcr2/gemma3_finetune"),
]


def main():
    # Collect raw texts once (shared across all models)
    print("="*60)
    print("Collecting raw texts for evaluation...")
    print("="*60)
    pubmed_texts = collect_pubmed_texts()
    wiki_texts   = collect_wiki_texts()
    c4_texts     = collect_c4_texts()
    texts_dict   = {"pubmed": pubmed_texts, "wiki": wiki_texts, "c4": c4_texts}

    all_results = {}

    for name, model_type, path in MODELS:
        print(f"\n{'='*60}")
        print(f"  {name}  —  {path}")
        print(f"{'='*60}")
        try:
            if model_type == "mamba":
                r = evaluate_mamba_on_texts(path, texts_dict)
            else:
                r = evaluate_hf_model(path, texts_dict)
            all_results[name] = {
                "pubmed_ppl": r["pubmed"]["ppl"],
                "wiki_ppl":   r["wiki"]["ppl"],
                "c4_ppl":     r["c4"]["ppl"],
                "pubmed_loss": r["pubmed"]["loss"],
                "wiki_loss":   r["wiki"]["loss"],
                "c4_loss":     r["c4"]["loss"],
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            all_results[name] = {
                "pubmed_ppl": None, "wiki_ppl": None, "c4_ppl": None,
                "error": str(e),
            }

        _save(all_results)  # incremental save

    # ── Summary table ──
    print("\n" + "="*75)
    print(f"{'Model':<22} {'PubMed PPL':>12} {'Wiki PPL':>12} {'C4 PPL':>12}")
    print("-"*75)
    for name, r in all_results.items():
        def fmt(v):
            return f"{v:>12.2f}" if v else "       ERROR"
        print(f"{name:<22} {fmt(r['pubmed_ppl'])} {fmt(r['wiki_ppl'])} {fmt(r['c4_ppl'])}")
    print("="*75)

    _save(all_results)
    print(f"\nResults saved to {RESULTS_JSON}")


if __name__ == "__main__":
    main()
