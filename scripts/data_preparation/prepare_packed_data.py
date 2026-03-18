#!/usr/bin/env python3
"""
Prepare packed CPT data: stream from MedRAG/pubmed, tokenize, pack, save.
Run this BEFORE training to ensure data is cached on disk.

Usage:
  python prepare_packed_data.py --max_samples 5000000
"""
# --- Project root resolution (auto-generated) ---
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')
_sys.path.insert(0, _PROJECT_ROOT)
_os.chdir(_PROJECT_ROOT)
# --- End project root resolution ---

import argparse
import os
import numpy as np
from transformers import AutoTokenizer

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max_samples", type=int, default=5_000_000)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--dataset", default="MedRAG/pubmed")
    p.add_argument("--out_dir", default="./data")
    args = p.parse_args()

    tag = f"packed_{args.max_length}_{args.max_samples // 1000}k"
    cache_dir = os.path.join(args.out_dir, f"pubmed_cpt_{tag}")

    if os.path.exists(os.path.join(cache_dir, "train_ids.npy")):
        print(f"[data] Packed data already exists at {cache_dir}")
        train_ids = np.load(os.path.join(cache_dir, "train_ids.npy"), mmap_mode='r')
        test_ids = np.load(os.path.join(cache_dir, "test_ids.npy"), mmap_mode='r')
        print(f"  Train: {len(train_ids):,} sequences, Test: {len(test_ids):,} sequences")
        print(f"  Total tokens: {len(train_ids) * args.max_length + len(test_ids) * args.max_length:,}")
        return

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")
    eos_id = tokenizer.eos_token_id or 0

    # Stream and collect texts
    print(f"[1/4] Streaming {args.max_samples:,} samples from {args.dataset}...")
    from datasets import load_dataset
    raw_stream = load_dataset(args.dataset, split="train", streaming=True)

    texts = []
    for i, ex in enumerate(raw_stream):
        if i >= args.max_samples:
            break
        content = (ex.get("content") or "").strip()
        if not content or len(content) < 50:
            continue
        title = (ex.get("title") or "").strip()
        text = f"{title}\n\n{content}" if title else content
        texts.append(text)
        if (i + 1) % 500_000 == 0:
            print(f"  {i+1:,} streamed, {len(texts):,} valid")

    print(f"  Total valid: {len(texts):,}")

    # Tokenize in batches and concatenate
    print(f"[2/4] Tokenizing {len(texts):,} texts...")
    all_ids = []
    batch_size = 10000
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(batch, add_special_tokens=False, truncation=False)
        for ids in encoded["input_ids"]:
            all_ids.extend(ids)
            all_ids.append(eos_id)
        done = min(start + batch_size, len(texts))
        if done % 500_000 < batch_size or done == len(texts):
            print(f"  {done:,} / {len(texts):,}")

    total_tokens = len(all_ids)
    print(f"  Total tokens: {total_tokens:,}")

    # Free texts memory
    del texts

    # Pack into fixed-length sequences as numpy array
    print(f"[3/4] Packing into {args.max_length}-token sequences...")
    n_seqs = total_tokens // args.max_length
    # Truncate to exact multiple
    all_ids = all_ids[:n_seqs * args.max_length]
    packed = np.array(all_ids, dtype=np.int32).reshape(n_seqs, args.max_length)
    del all_ids
    print(f"  {n_seqs:,} packed sequences ({n_seqs * args.max_length:,} tokens, 0% padding)")

    # Train/test split (2% test)
    np.random.seed(42)
    indices = np.random.permutation(n_seqs)
    test_size = max(1000, int(n_seqs * 0.02))
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    train_ids = packed[train_idx]
    test_ids = packed[test_idx]

    # Save
    print(f"[4/4] Saving to {cache_dir}...")
    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, "train_ids.npy"), train_ids)
    np.save(os.path.join(cache_dir, "test_ids.npy"), test_ids)
    print(f"  Train: {len(train_ids):,}, Test: {len(test_ids):,}")
    print(f"  Files: {os.path.getsize(os.path.join(cache_dir, 'train_ids.npy')) / 1e9:.2f} GB (train)")
    print("Done!")


if __name__ == "__main__":
    main()
