#!/usr/bin/env python3
"""
Prepare PubMed + C4/Wiki mixed datasets at various ratios for ablation study.
"""
# --- Project root resolution (auto-generated) ---
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')
_sys.path.insert(0, _PROJECT_ROOT)
_os.chdir(_PROJECT_ROOT)
# --- End project root resolution ---

import os
from datasets import load_from_disk, load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer


def stream_and_tokenize(dataset_iter, tokenizer, n_needed, max_length=1024, source_name="data"):
    """Stream from HF dataset, tokenize, return lists of input_ids and attention_mask."""
    input_ids_list = []
    attention_masks_list = []
    count = 0
    for ex in dataset_iter:
        text = ex.get("text", "").strip()
        if not text or len(text) < 100:
            continue
        enc = tokenizer(text, truncation=True, max_length=max_length,
                        padding="max_length", return_tensors="np")
        ids = enc["input_ids"][0].tolist()
        mask = enc["attention_mask"][0].tolist()
        if sum(mask) < 128:
            continue
        input_ids_list.append(ids)
        attention_masks_list.append(mask)
        count += 1
        if count % 10000 == 0:
            print(f"  [{source_name}] {count:,} / {n_needed:,}")
        if count >= n_needed:
            break
    print(f"  [{source_name}] Collected {count:,} sequences")
    return input_ids_list, attention_masks_list


def prepare_mix(pubmed_train, pubmed_test, tokenizer, c4_pct, wiki_pct, max_length=1024):
    """Prepare a PubMed + c4_pct% C4 + wiki_pct% Wiki dataset."""
    if c4_pct > 0 and wiki_pct > 0:
        tag = f"c4{c4_pct}_wiki{wiki_pct}"
    elif c4_pct > 0:
        tag = f"c4_only_{c4_pct}pct"
    else:
        tag = f"wiki_only_{wiki_pct}pct"

    out_dir = os.path.join(_PROJECT_ROOT, f"data/pubmed_mix_{tag}")
    if os.path.exists(os.path.join(out_dir, "train", "state.json")):
        print(f"Already exists: {out_dir}")
        return out_dir

    n_c4 = int(len(pubmed_train) * c4_pct / 100) if c4_pct > 0 else 0
    n_wiki = int(len(pubmed_train) * wiki_pct / 100) if wiki_pct > 0 else 0

    print(f"\n{'='*60}")
    print(f"Preparing: PubMed + {c4_pct}%C4 + {wiki_pct}%Wiki")
    print(f"  PubMed: {len(pubmed_train):,}, C4: {n_c4:,}, Wiki: {n_wiki:,}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    extra_datasets = []

    # Stream C4
    if n_c4 > 0:
        c4_iter = load_dataset("allenai/c4", "en", split="train", streaming=True)
        c4_ids, c4_masks = stream_and_tokenize(c4_iter, tokenizer, n_c4, max_length, "C4")
        c4_ds = Dataset.from_dict({"input_ids": c4_ids, "attention_mask": c4_masks})
        extra_datasets.append(("C4", c4_ds))

    # Stream Wikipedia
    if n_wiki > 0:
        wiki_iter = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        wiki_ids, wiki_masks = stream_and_tokenize(wiki_iter, tokenizer, n_wiki, max_length, "Wiki")
        wiki_ds = Dataset.from_dict({"input_ids": wiki_ids, "attention_mask": wiki_masks})
        extra_datasets.append(("Wiki", wiki_ds))

    # Concatenate
    all_ds = [pubmed_train] + [ds for _, ds in extra_datasets]
    combined_train = concatenate_datasets(all_ds)
    combined_train = combined_train.shuffle(seed=42)

    total = len(combined_train)
    print(f"\n  Combined train: {total:,} sequences")
    print(f"    PubMed: {len(pubmed_train):,} ({100*len(pubmed_train)/total:.1f}%)")
    for name, ds in extra_datasets:
        print(f"    {name}: {len(ds):,} ({100*len(ds)/total:.1f}%)")

    os.makedirs(out_dir, exist_ok=True)
    combined_train.save_to_disk(os.path.join(out_dir, "train"))
    pubmed_test.save_to_disk(os.path.join(out_dir, "test"))
    print(f"  Saved to {out_dir}")
    return out_dir


def main():
    print("Loading base PubMed data...")
    pubmed_train = load_from_disk(os.path.join(_PROJECT_ROOT, "data/pubmed_medline_tokenized_1024/train"))
    pubmed_test = load_from_disk(os.path.join(_PROJECT_ROOT, "data/pubmed_medline_tokenized_1024/test"))
    print(f"  PubMed train: {len(pubmed_train):,}, test: {len(pubmed_test):,}")

    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")

    # 5 experiments:
    configs = [
        (5,  0),   # A: PubMed + 5%C4
        (20, 0),   # B: PubMed + 20%C4
        (30, 0),   # C: PubMed + 30%C4
        (5,  5),   # D: PubMed + 5%C4 + 5%Wiki
        (20, 10),  # E: PubMed + 20%C4 + 10%Wiki
    ]

    for c4_pct, wiki_pct in configs:
        prepare_mix(pubmed_train, pubmed_test, tokenizer, c4_pct, wiki_pct)

    print("\nAll datasets prepared!")


if __name__ == "__main__":
    main()
