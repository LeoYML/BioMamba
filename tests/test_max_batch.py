#!/usr/bin/env python3
"""Test max batch size for each model size on H100 80GB."""
# --- Project root resolution (auto-generated) ---
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')
_sys.path.insert(0, _PROJECT_ROOT)
_os.chdir(_PROJECT_ROOT)
# --- End project root resolution ---

import torch
import gc

from ft_biomamba.model import load_model, load_tokenizer

MODELS = [
    ("370m", "state-spaces/mamba2-370m"),
    ("780m", "state-spaces/mamba2-780m"),
    ("1.3b", "state-spaces/mamba2-1.3b"),
    ("2.7b", "state-spaces/mamba2-2.7b"),
]

SEQ_LEN = 1024
device = "cuda"
tokenizer = load_tokenizer()

for name, path in MODELS:
    print(f"\n=== Testing {name} ===")
    torch.cuda.empty_cache()
    gc.collect()

    model = load_model(path, device=device)
    model.train()

    # Binary search for max batch size
    lo, hi = 1, 128
    best = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        torch.cuda.empty_cache()
        try:
            dummy = torch.randint(0, 50000, (mid, SEQ_LEN), device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(dummy)
                loss = out.logits.mean()  # fake loss
            loss.backward()
            model.zero_grad()
            del dummy, out, loss
            torch.cuda.empty_cache()
            best = mid
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"  batch={mid} OK, peak={mem:.1f}GB")
            torch.cuda.reset_peak_memory_stats()
            lo = mid + 1
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower():
                del dummy
                torch.cuda.empty_cache()
                hi = mid - 1
            else:
                raise

    # Suggest: use ~80% of max to leave room for optimizer states
    suggested = max(1, int(best * 0.7))
    accum = max(1, 32 // suggested)
    eff = suggested * accum
    print(f"  >>> {name}: max_batch={best}, suggested_batch={suggested}, accum={accum}, effective={eff}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
