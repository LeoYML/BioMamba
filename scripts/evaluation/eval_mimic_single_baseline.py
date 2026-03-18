#!/usr/bin/env python3
"""
Evaluate a single external model on MIMIC-IV completion + discharge.
Called as subprocess by eval_mimic_baselines.py for CUDA isolation.
Uses batched generation for GPU utilization.
"""
# --- Project root resolution (auto-generated) ---
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')
_sys.path.insert(0, _PROJECT_ROOT)
_os.chdir(_PROJECT_ROOT)
# --- End project root resolution ---

import argparse, json, os, sys, torch, warnings
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, os.path.dirname(__file__))
from ft_biomamba.mimic_data import load_mimic_notes, split_mimic_data, prepare_completion_data, prepare_discharge_data
from ft_biomamba.mimic_prompts import format_completion_prompt, format_discharge_prompt
from ft_biomamba.model import load_tokenizer as load_mamba_tokenizer
from rouge_score import rouge_scorer

DEVICE = "cuda"
MAX_NEW_TOKENS = 128
MAX_SAMPLES = 500
BATCH_SIZE = 16  # batch generation


def load_hf_model(model_id):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # critical for batched generation

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(DEVICE)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_batch(model, tokenizer, prompts, max_new_tokens=128, max_length=1024):
    """Generate text for a batch of prompts."""
    model_max = getattr(model.config, "max_position_embeddings", max_length)
    effective_max = min(max_length, model_max) if model_max else max_length
    trunc_len = max(effective_max - max_new_tokens, 128)

    inputs = tokenizer(
        prompts, return_tensors="pt", truncation=True,
        max_length=trunc_len, padding=True,
    )
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    try:
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    except Exception as e:
        print(f"    [warn] Batch generation error: {e}")
        return [""] * len(prompts)

    results = []
    for i in range(len(prompts)):
        input_len = (attention_mask[i] == 1).sum().item()
        new_ids = outputs[i][input_ids.shape[1]:]
        # Also skip pad tokens in output
        new_ids = new_ids[new_ids != tokenizer.pad_token_id]
        if new_ids.numel() == 0:
            results.append("")
        else:
            results.append(tokenizer.decode(new_ids, skip_special_tokens=True))
    return results


def eval_task(model, tokenizer, samples, prompt_fn, max_new_tokens, scorer, task_name):
    """Evaluate a task with batched generation."""
    all_r1, all_r2, all_rL = [], [], []
    prompts_batch = []
    refs_batch = []
    indices = []

    def flush_batch():
        if not prompts_batch:
            return
        generated_list = generate_batch(model, tokenizer, prompts_batch, max_new_tokens)
        for gen, ref, idx in zip(generated_list, refs_batch, indices):
            if gen and ref:
                sc = scorer.score(ref, gen)
                all_r1.append(sc["rouge1"].fmeasure)
                all_r2.append(sc["rouge2"].fmeasure)
                all_rL.append(sc["rougeL"].fmeasure)
            else:
                all_r1.append(0.0)
                all_r2.append(0.0)
                all_rL.append(0.0)
            if idx < 3:
                print(f"  [{idx}] R1={all_r1[-1]:.3f} gen: {gen[:100]}...")
        prompts_batch.clear()
        refs_batch.clear()
        indices.clear()

    for i, sample in enumerate(tqdm(samples, desc=task_name)):
        prompt, ref = prompt_fn(sample)
        prompts_batch.append(prompt)
        refs_batch.append(ref)
        indices.append(i)

        if len(prompts_batch) >= BATCH_SIZE:
            flush_batch()

    flush_batch()  # remaining

    return {
        "rouge1": float(np.mean(all_r1)) if all_r1 else 0.0,
        "rouge2": float(np.mean(all_r2)) if all_r2 else 0.0,
        "rougeL": float(np.mean(all_rL)) if all_rL else 0.0,
        "total": len(samples),
        "valid": sum(1 for r in all_r1 if r > 0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    global BATCH_SIZE
    BATCH_SIZE = args.batch_size

    print(f"Loading MIMIC data...")
    notes = load_mimic_notes(os.path.join(_PROJECT_ROOT, "data/mimic-iv-note/2.2/note"))
    _, test_df = split_mimic_data(notes, test_ratio=0.1, seed=42)

    mamba_tokenizer = load_mamba_tokenizer()
    comp_samples = prepare_completion_data(test_df, mamba_tokenizer, max_samples=MAX_SAMPLES)
    disc_samples = prepare_discharge_data(test_df, max_samples=MAX_SAMPLES)

    print(f"Loading model: {args.model_name} ({args.model_id}), batch_size={BATCH_SIZE}")
    model, tokenizer = load_hf_model(args.model_id)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    # Completion
    print(f"Completion eval ({len(comp_samples)} samples, bs={BATCH_SIZE})...")
    comp_results = eval_task(
        model, tokenizer, comp_samples,
        lambda s: (format_completion_prompt(s["prefix_text"]), s["reference_text"]),
        MAX_NEW_TOKENS, scorer, "Completion"
    )

    # Discharge
    print(f"Discharge eval ({len(disc_samples)} samples, bs={BATCH_SIZE})...")
    disc_results = eval_task(
        model, tokenizer, disc_samples,
        lambda s: (format_discharge_prompt(s["context"], max_ctx_chars=4000), s["reference"]),
        min(MAX_NEW_TOKENS, 256), scorer, "Discharge"
    )

    results = {
        "name": args.model_name,
        "model_id": args.model_id,
        "completion": comp_results,
        "discharge": disc_results,
    }

    c = results["completion"]
    d = results["discharge"]
    print(f"\n{args.model_name}: C-R1={c['rouge1']*100:.2f} C-R2={c['rouge2']*100:.2f} C-RL={c['rougeL']*100:.2f} "
          f"D-R1={d['rouge1']*100:.2f} D-R2={d['rouge2']*100:.2f} D-RL={d['rougeL']*100:.2f}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
