#!/usr/bin/env python3
"""
Perplexity evaluation on PubMed-MEDLINE for ALL models.

For causal LMs (Mamba, BioGPT): standard next-token prediction loss.
For encoder-decoder (T5): decoder cross-entropy with text as both input and target.

NOTE: T5 perplexity is not directly comparable to causal LM perplexity because
the T5 encoder sees the full text bidirectionally. We report both but flag the
architectural difference.
"""
# --- Project root resolution (auto-generated) ---
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')
_sys.path.insert(0, _PROJECT_ROOT)
_os.chdir(_PROJECT_ROOT)
# --- End project root resolution ---

import json
import math
import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from datasets import load_from_disk

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_SAMPLES = 500
MAX_LEN = 512


def compute_causal_ppl(model, tokenizer, texts, name):
    """Perplexity for causal LMs (Mamba, BioGPT)."""
    total_loss = 0.0
    total_tokens = 0

    for text in tqdm(texts, desc=name):
        enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=MAX_LEN)
        input_ids = enc['input_ids'].to(DEVICE)
        if input_ids.shape[1] < 2:
            continue

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

        # Shift for causal LM
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        total_loss += loss.item()
        total_tokens += shift_labels.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl, total_tokens


def compute_mamba_ppl(model_path, texts, name):
    """Perplexity for Mamba2 models."""
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = MambaLMHeadModel.from_pretrained(model_path, device=str(DEVICE), dtype=torch.float32)
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for text in tqdm(texts, desc=name):
        enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=MAX_LEN)
        input_ids = enc['input_ids'].to(DEVICE)
        if input_ids.shape[1] < 2:
            continue

        with torch.no_grad():
            out = model(input_ids)
            logits = out.logits if hasattr(out, 'logits') else out[0]

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        total_loss += loss.item()
        total_tokens += shift_labels.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)

    del model
    torch.cuda.empty_cache()
    return avg_loss, ppl, total_tokens


def compute_t5_ppl(model_path, texts, name):
    """Perplexity for T5 encoder-decoder models.

    Uses text as both encoder input and decoder target.
    This measures reconstruction ability, not pure LM perplexity.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(DEVICE)
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for text in tqdm(texts, desc=name):
        enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=MAX_LEN)
        input_ids = enc['input_ids'].to(DEVICE)
        if input_ids.shape[1] < 2:
            continue

        with torch.no_grad():
            # Use text as both input and target
            outputs = model(input_ids=input_ids, labels=input_ids)
            # outputs.loss is mean cross-entropy over target tokens
            n_tokens = input_ids.numel()
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)

    del model, tokenizer
    torch.cuda.empty_cache()
    return avg_loss, ppl, total_tokens


def main():
    # Load PubMed test data (same as previous evaluation)
    print("Loading PubMed data...")
    try:
        ds = load_from_disk(os.path.join(_PROJECT_ROOT, 'data/pubmed_medline_cpt/test'))
        texts = [ds[i]['text'] for i in range(min(NUM_SAMPLES, len(ds)))]
        print(f"PubMed test: {len(ds)} samples")
    except Exception:
        # Fallback: load from HuggingFace
        from datasets import load_dataset
        ds = load_dataset("pubmed_qa", "pqa_labeled", split="train")
        texts = [ex['context']['contexts'][0] for ex in ds if ex['context']['contexts']]
        texts = texts[:NUM_SAMPLES]

    print(f"Using {len(texts)} samples for perplexity evaluation\n")

    # Load existing results
    out_path = os.path.join(_PROJECT_ROOT, 'evaluation_results/eval_perplexity_pubmed_20260226.json')
    try:
        with open(out_path) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results")
    except Exception:
        results = {}

    # ── Mamba models (skip if already done) ──
    mamba_models = [
        ('Mamba2-130m-base', 'state-spaces/mamba2-130m'),
        ('BioMamba-CPT', os.path.join(_PROJECT_ROOT, 'checkpoints/biomamba_cpt_mamba2-130m/best_model')),
        ('BioMamba-CPT+SFT', os.path.join(_PROJECT_ROOT, 'checkpoints/clean_mixed/biomamba_sft_mamba2-130m_full/best_model')),
    ]

    for name, path in mamba_models:
        if name in results:
            print(f"=== {name} === (cached: ppl={results[name]['perplexity']:.2f})")
            continue
        print(f"\n=== {name} ===")
        loss, ppl, tokens = compute_mamba_ppl(path, texts, name)
        results[name] = {'loss': loss, 'perplexity': ppl, 'tokens': tokens, 'type': 'causal'}
        print(f"  loss={loss:.4f}  ppl={ppl:.2f}  tokens={tokens}")

    # ── BioGPT models (skip if already done) ──
    biogpt_models = [
        ('BioGPT', 'microsoft/biogpt'),
        ('BioGPT-large', 'microsoft/biogpt-large'),
    ]

    for name, path in biogpt_models:
        if name in results:
            print(f"=== {name} === (cached: ppl={results[name]['perplexity']:.2f})")
            continue
        print(f"\n=== {name} ===")
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path).to(DEVICE)
        model.eval()
        loss, ppl, tokens = compute_causal_ppl(model, tokenizer, texts, name)
        results[name] = {'loss': loss, 'perplexity': ppl, 'tokens': tokens, 'type': 'causal'}
        print(f"  loss={loss:.4f}  ppl={ppl:.2f}  tokens={tokens}")
        del model, tokenizer
        torch.cuda.empty_cache()

    # ── T5 models (new) ──
    t5_models = [
        ('flan-t5-small', 'google/flan-t5-small'),
        ('flan-t5-base', 'google/flan-t5-base'),
        ('flan-t5-large', 'google/flan-t5-large'),
        ('medical-qa-t5-lora', 'Adilbai/medical-qa-t5-lora'),
    ]

    for name, path in t5_models:
        if name in results:
            print(f"=== {name} === (cached: ppl={results[name]['perplexity']:.2f})")
            continue
        print(f"\n=== {name} ===")
        loss, ppl, tokens = compute_t5_ppl(path, texts, name)
        results[name] = {'loss': loss, 'perplexity': ppl, 'tokens': tokens, 'type': 'encoder-decoder'}
        print(f"  loss={loss:.4f}  ppl={ppl:.2f}  tokens={tokens}")

    # Save
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print(f"{'Model':<25} {'Type':<18} {'Loss':>8} {'PPL':>10} {'Tokens':>10}")
    print('-'*80)

    # Sort by PPL within each type group
    causal = [(n, r) for n, r in results.items() if r.get('type', 'causal') == 'causal']
    enc_dec = [(n, r) for n, r in results.items() if r.get('type') == 'encoder-decoder']

    print("  ── Causal LMs ──")
    for name, r in sorted(causal, key=lambda x: x[1]['perplexity']):
        print(f"  {name:<23} {'Causal':<18} {r['loss']:>8.4f} {r['perplexity']:>10.2f} {r['tokens']:>10,}")

    print("  ── Encoder-Decoder ──")
    for name, r in sorted(enc_dec, key=lambda x: x[1]['perplexity']):
        print(f"  {name:<23} {'Enc-Dec':<18} {r['loss']:>8.4f} {r['perplexity']:>10.2f} {r['tokens']:>10,}")

    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
