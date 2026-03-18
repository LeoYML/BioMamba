#!/usr/bin/env python3
"""
Take existing pubmed_medline_tokenized_1024 (424K) + add 10% general corpus (C4).
Saves as HF dataset with input_ids + attention_mask.
"""
# --- Project root resolution (auto-generated) ---
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')
_sys.path.insert(0, _PROJECT_ROOT)
_os.chdir(_PROJECT_ROOT)
# --- End project root resolution ---

import os
import numpy as np
from datasets import load_from_disk, load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer

def main():
    out_dir = os.path.join(_PROJECT_ROOT, "data/pubmed_mixed_10pct_general")
    if os.path.exists(os.path.join(out_dir, "train")):
        print(f"Already exists: {out_dir}")
        return

    # Load existing PubMed data
    print("Loading existing pubmed_medline_tokenized_1024...")
    train_ds = load_from_disk(os.path.join(_PROJECT_ROOT, "data/pubmed_medline_tokenized_1024/train"))
    test_ds = load_from_disk(os.path.join(_PROJECT_ROOT, "data/pubmed_medline_tokenized_1024/test"))
    print(f"  PubMed train: {len(train_ds):,}, test: {len(test_ds):,}")

    # Calculate how many general corpus sequences we need (10% of train)
    n_general = int(len(train_ds) * 0.10)
    print(f"  Need {n_general:,} general corpus sequences")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")
    max_length = 1024
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # Stream from C4 and tokenize
    print(f"Streaming {n_general:,} sequences from C4...")
    c4 = load_dataset("allenai/c4", "en", split="train", streaming=True)

    general_input_ids = []
    general_attention_masks = []
    count = 0
    for ex in c4:
        text = ex.get("text", "").strip()
        if not text or len(text) < 100:
            continue
        
        enc = tokenizer(text, truncation=True, max_length=max_length,
                       padding="max_length", return_tensors="np")
        ids = enc["input_ids"][0].tolist()
        mask = enc["attention_mask"][0].tolist()
        
        # Only keep if at least 128 real tokens
        if sum(mask) < 128:
            continue
        
        general_input_ids.append(ids)
        general_attention_masks.append(mask)
        count += 1
        
        if count % 10000 == 0:
            print(f"  {count:,} / {n_general:,}")
        
        if count >= n_general:
            break

    print(f"  Collected {count:,} general corpus sequences")

    # Create general dataset
    general_ds = Dataset.from_dict({
        "input_ids": general_input_ids,
        "attention_mask": general_attention_masks,
    })

    # Concatenate
    from datasets import concatenate_datasets
    combined_train = concatenate_datasets([train_ds, general_ds])
    combined_train = combined_train.shuffle(seed=42)
    
    print(f"  Combined train: {len(combined_train):,} sequences")
    print(f"    PubMed: {len(train_ds):,} ({100*len(train_ds)/len(combined_train):.1f}%)")
    print(f"    General: {count:,} ({100*count/len(combined_train):.1f}%)")

    # Save
    os.makedirs(out_dir, exist_ok=True)
    combined_train.save_to_disk(os.path.join(out_dir, "train"))
    test_ds.save_to_disk(os.path.join(out_dir, "test"))
    print(f"Saved to {out_dir}")

if __name__ == "__main__":
    main()
