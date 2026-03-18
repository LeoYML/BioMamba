#!/usr/bin/env python3
"""
Prepare PubMed + Wiki-only datasets (no C4) at 10%, 20%, 30% Wiki ratios.
"""
# --- Project root resolution (auto-generated) ---
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')
_sys.path.insert(0, _PROJECT_ROOT)
_os.chdir(_PROJECT_ROOT)
# --- End project root resolution ---

import os
import numpy as np
from datasets import load_from_disk, load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer


def prepare_wiki_mix(pubmed_train, pubmed_test, tokenizer, wiki_ratio, max_length=1024):
    out_dir = os.path.join(_PROJECT_ROOT, f"data/pubmed_wiki_only_{wiki_ratio}pct")
    if os.path.exists(os.path.join(out_dir, "train", "state.json")):
        print(f"Already exists: {out_dir}")
        return out_dir

    n_wiki = int(len(pubmed_train) * wiki_ratio / 100)
    print(f"\n{'='*60}")
    print(f"Preparing PubMed + {wiki_ratio}% Wikipedia (no C4)")
    print(f"  PubMed: {len(pubmed_train):,}, Wiki needed: {n_wiki:,}")
    print(f"{'='*60}")

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # Stream from Wikipedia
    print(f"Streaming {n_wiki:,} sequences from Wikipedia...")
    wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

    wiki_input_ids = []
    wiki_attention_masks = []
    count = 0
    for ex in wiki:
        text = ex.get("text", "").strip()
        if not text or len(text) < 100:
            continue

        enc = tokenizer(text, truncation=True, max_length=max_length,
                        padding="max_length", return_tensors="np")
        ids = enc["input_ids"][0].tolist()
        mask = enc["attention_mask"][0].tolist()

        if sum(mask) < 128:
            continue

        wiki_input_ids.append(ids)
        wiki_attention_masks.append(mask)
        count += 1

        if count % 10000 == 0:
            print(f"  {count:,} / {n_wiki:,}")

        if count >= n_wiki:
            break

    print(f"  Collected {count:,} Wikipedia sequences")

    wiki_ds = Dataset.from_dict({
        "input_ids": wiki_input_ids,
        "attention_mask": wiki_attention_masks,
    })

    combined_train = concatenate_datasets([pubmed_train, wiki_ds])
    combined_train = combined_train.shuffle(seed=42)

    print(f"  Combined train: {len(combined_train):,} sequences")
    print(f"    PubMed: {len(pubmed_train):,} ({100*len(pubmed_train)/len(combined_train):.1f}%)")
    print(f"    Wiki:   {count:,} ({100*count/len(combined_train):.1f}%)")

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

    for ratio in [10, 20, 30]:
        prepare_wiki_mix(pubmed_train, pubmed_test, tokenizer, ratio)

    print("\nAll datasets prepared!")


if __name__ == "__main__":
    main()
