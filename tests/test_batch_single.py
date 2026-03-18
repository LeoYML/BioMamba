#!/usr/bin/env python3
"""Test max training batch size for a single model (with optimizer states)."""
# --- Project root resolution (auto-generated) ---
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')
_sys.path.insert(0, _PROJECT_ROOT)
_os.chdir(_PROJECT_ROOT)
# --- End project root resolution ---

import sys
import torch
import gc
from ft_biomamba.model import load_model, load_tokenizer

model_path = sys.argv[1]
SEQ_LEN = 1024
device = "cuda"

torch.cuda.empty_cache()
gc.collect()

model = load_model(model_path, device=device)
model.train()

# Create optimizer to account for optimizer memory
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scaler = torch.amp.GradScaler("cuda")

sizes_to_try = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]
best = 0

for bs in sizes_to_try:
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    try:
        dummy = torch.randint(0, 50000, (bs, SEQ_LEN), device=device)
        labels = torch.randint(0, 50000, (bs, SEQ_LEN), device=device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(dummy)
            loss = torch.nn.functional.cross_entropy(
                out.logits[:, :-1].reshape(-1, out.logits.size(-1)),
                labels[:, 1:].reshape(-1)
            )
        loss.backward()
        optimizer.step()

        peak = torch.cuda.max_memory_allocated() / 1e9
        util = peak / 80 * 100
        print(f"  batch={bs:>3d}  peak={peak:.1f}GB  util={util:.0f}%  OK")
        best = bs

        del dummy, labels, out, loss
        torch.cuda.empty_cache()
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower():
            print(f"  batch={bs:>3d}  OOM")
            break
        raise

# Suggest using ~85% of max
suggested = best
accum = max(1, 32 // suggested)
eff = suggested * accum
print(f"\n  >>> max_batch={best}, recommended: batch={suggested}, accum={accum}, effective={eff}")
