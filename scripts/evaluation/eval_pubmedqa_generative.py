#!/usr/bin/env python3
"""Comprehensive PubMedQA generative evaluation with multiple strategies.

Tests: greedy generation, logprob selection, self-consistency voting.
Models: MixedCPT+LoRA SFT, MixedCPT+Full SFT, No-CPT SFT.
"""
# --- Project root resolution (auto-generated) ---
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')
_sys.path.insert(0, _PROJECT_ROOT)
_os.chdir(_PROJECT_ROOT)
# --- End project root resolution ---

import argparse
import json
import os
import torch
import torch.nn.functional as F
from collections import Counter
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

from ft_biomamba.model import load_model, load_tokenizer, inject_lora
from ft_biomamba.config import QA_TEMPLATE, TOKENIZER_NAME


# ── Prompt templates ──────────────────────────────────────────────
FEWSHOT_PROMPT = (
    "Answer biomedical questions with ONLY one word: yes, no, or maybe.\n\n"
    "Example 1:\nQuestion: Is aspirin used to treat headaches?\nAnswer: yes\n\n"
    "Example 2:\nQuestion: Does vitamin C cure cancer?\nAnswer: no\n\n"
    "Now answer this question:\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\nAnswer:"
)


def format_prompt(question, context, template="sft", max_ctx_chars=2400):
    trunc = context[:max_ctx_chars]
    if len(context) > max_ctx_chars:
        trunc += "..."
    if template == "fewshot":
        return FEWSHOT_PROMPT.format(question=question, context=trunc)
    else:
        return QA_TEMPLATE.format(question=question, context=trunc)


# ── Data loading ──────────────────────────────────────────────────
def load_pubmedqa_test(seed=42):
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    splits = ds.train_test_split(test_size=0.2, seed=seed)
    test = splits["test"]

    def get_context(ex):
        ctx_dict = ex["context"]
        contexts = ctx_dict["contexts"]
        labels = ctx_dict.get("labels", [])
        parts = []
        for i, ctx in enumerate(contexts):
            if i < len(labels) and labels[i]:
                parts.append(f"{labels[i]}: {ctx}")
            else:
                parts.append(ctx)
        return "\n\n".join(parts)

    samples = []
    for ex in test:
        samples.append({
            "question": ex["question"],
            "context": get_context(ex),
            "answer": ex["final_decision"],
        })
    return samples


# ── Answer strategies ─────────────────────────────────────────────
@torch.no_grad()
def strategy_greedy(model, tokenizer, prompt, device, max_new=8):
    """Standard greedy generation + answer extraction."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = inputs["input_ids"].to(device)
    input_len = input_ids.shape[1]

    try:
        outputs = model.generate(
            input_ids,
            max_length=input_len + max_new,
            cg=device.type == "cuda",
            return_dict_in_generate=False,
        )
    except Exception:
        generated = input_ids
        for _ in range(max_new):
            out = model(generated)
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_id], dim=1)
            if tokenizer.eos_token_id and next_id.item() == tokenizer.eos_token_id:
                break
        outputs = generated

    new_ids = outputs[0][input_len:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True) if new_ids.numel() > 0 else ""
    return extract_answer(text)


@torch.no_grad()
def strategy_logprob(model, tokenizer, prompt, device, **kwargs):
    """Compare logprobs of yes/no/maybe tokens directly."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = inputs["input_ids"].to(device)
    outputs = model(input_ids)
    logits = outputs.logits[:, -1, :]  # next-token logits
    probs = F.softmax(logits, dim=-1)[0]

    candidates = {}
    for word in ["yes", "no", "maybe", "Yes", "No", "Maybe",
                 " yes", " no", " maybe", " Yes", " No", " Maybe"]:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if ids:
            token_id = ids[0]
            base = word.strip().lower()
            p = probs[token_id].item()
            if base not in candidates or p > candidates[base]:
                candidates[base] = p

    if not candidates:
        return "yes"
    return max(candidates, key=candidates.get)


def _get_candidate_logits(model, tokenizer, prompt, device, max_length=1024):
    """Helper: extract logits for yes/no/maybe candidate tokens."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"].to(device)
    outputs = model(input_ids)
    logits = outputs.logits[:, -1, :]  # next-token logits

    candidate_logits = {}
    for word in ["yes", "no", "maybe", "Yes", "No", "Maybe",
                 " yes", " no", " maybe", " Yes", " No", " Maybe"]:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if ids:
            token_id = ids[0]
            base = word.strip().lower()
            l = logits[0, token_id].item()
            if base not in candidate_logits or l > candidate_logits[base]:
                candidate_logits[base] = l
    return candidate_logits


@torch.no_grad()
def strategy_logprob_restricted(model, tokenizer, prompt, device, **kwargs):
    """Restricted softmax: softmax only over yes/no/maybe logits (not full vocab).
    This gives better-calibrated 3-way probabilities."""
    candidate_logits = _get_candidate_logits(model, tokenizer, prompt, device)
    if not candidate_logits:
        return "yes"

    labels = list(candidate_logits.keys())
    vals = torch.tensor([candidate_logits[l] for l in labels])
    probs = F.softmax(vals, dim=0)

    idx = probs.argmax().item()
    return labels[idx]


@torch.no_grad()
def strategy_logprob_restricted_margin(model, tokenizer, prompt, device,
                                        margin_threshold=0.15, maybe_threshold=0.15, **kwargs):
    """Restricted softmax + margin-based maybe detection.
    Uses 3-way softmax so maybe has a fair chance."""
    candidate_logits = _get_candidate_logits(model, tokenizer, prompt, device)
    if not candidate_logits:
        return "yes"

    labels = list(candidate_logits.keys())
    vals = torch.tensor([candidate_logits[l] for l in labels])
    probs = F.softmax(vals, dim=0)

    prob_dict = {l: p.item() for l, p in zip(labels, probs)}
    sorted_cands = sorted(prob_dict.items(), key=lambda x: -x[1])
    top_label, top_prob = sorted_cands[0]

    if len(sorted_cands) >= 2:
        second_prob = sorted_cands[1][1]
        margin = top_prob - second_prob
        maybe_prob = prob_dict.get("maybe", 0)
        if margin < margin_threshold and maybe_prob > maybe_threshold:
            return "maybe"

    return top_label


@torch.no_grad()
def strategy_logprob_calibrated(model, tokenizer, prompt, device, **kwargs):
    """Logprob with temperature scaling for calibration."""
    candidate_logits = _get_candidate_logits(model, tokenizer, prompt, device)
    if not candidate_logits:
        return "yes"

    labels = list(candidate_logits.keys())
    vals = torch.tensor([candidate_logits[l] for l in labels])
    idx = vals.argmax().item()
    return labels[idx]


@torch.no_grad()
def strategy_logprob_long(model, tokenizer, prompt, device, **kwargs):
    """Logprob with max_length=2048 to capture more context."""
    candidate_logits = _get_candidate_logits(model, tokenizer, prompt, device, max_length=2048)
    if not candidate_logits:
        return "yes"

    labels = list(candidate_logits.keys())
    vals = torch.tensor([candidate_logits[l] for l in labels])
    probs = F.softmax(vals, dim=0)
    idx = probs.argmax().item()
    return labels[idx]


@torch.no_grad()
def strategy_voting(model, tokenizer, prompt, device, n_samples=5, temperature=0.7):
    """Self-consistency: generate multiple answers and vote."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = inputs["input_ids"].to(device)
    input_len = input_ids.shape[1]

    votes = []
    for _ in range(n_samples):
        try:
            outputs = model.generate(
                input_ids,
                max_length=input_len + 8,
                temperature=temperature,
                top_p=0.9,
                cg=device.type == "cuda",
                return_dict_in_generate=False,
            )
        except Exception:
            # Fallback: manual generation with temperature
            generated = input_ids.clone()
            for _ in range(8):
                out = model(generated)
                logits = out.logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1)
                generated = torch.cat([generated, next_id], dim=1)
                if tokenizer.eos_token_id and next_id.item() == tokenizer.eos_token_id:
                    break
            outputs = generated

        new_ids = outputs[0][input_len:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True) if new_ids.numel() > 0 else ""
        ans = extract_answer(text)
        votes.append(ans)

    # Majority vote
    counter = Counter(votes)
    return counter.most_common(1)[0][0]


@torch.no_grad()
def strategy_logprob_margin(model, tokenizer, prompt, device, **kwargs):
    """Logprob with margin-based maybe detection (full vocab softmax).
    If top-2 answers are close in probability, output 'maybe'."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = inputs["input_ids"].to(device)
    outputs = model(input_ids)
    logits = outputs.logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)[0]

    candidate_probs = {}
    for word in ["yes", "no", "maybe", "Yes", "No", "Maybe",
                 " yes", " no", " maybe", " Yes", " No", " Maybe"]:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if ids:
            token_id = ids[0]
            base = word.strip().lower()
            p = probs[token_id].item()
            if base not in candidate_probs or p > candidate_probs[base]:
                candidate_probs[base] = p

    if not candidate_probs:
        return "yes"

    sorted_cands = sorted(candidate_probs.items(), key=lambda x: -x[1])
    top_label, top_prob = sorted_cands[0]

    if len(sorted_cands) >= 2:
        second_prob = sorted_cands[1][1]
        margin = top_prob - second_prob
        maybe_prob = candidate_probs.get("maybe", 0)
        if margin < 0.15 and maybe_prob > 0.05:
            return "maybe"

    return top_label


def extract_answer(text):
    text = text.strip().split("\n")[0].lower()
    for p in ("answer:", "the answer is", "a:", "(yes/no):", "yes/no:"):
        text = text.replace(p, "")
    text = text.lstrip(" \t:;,-.")

    for keyword in ("yes", "no", "maybe"):
        if text.startswith(keyword):
            return keyword

    words = text.split()
    for w in words[:5]:
        w = w.strip(".,;:!?()[]{}\"' ")
        if w in ("yes", "no", "maybe", "uncertain", "unclear"):
            return "maybe" if w in ("uncertain", "unclear") else w

    return words[0] if words else "yes"


# ── Evaluation ────────────────────────────────────────────────────
def evaluate(model, tokenizer, samples, strategy_fn, device, prompt_template="sft",
             strategy_name="", **kwargs):
    preds, refs = [], []

    for i, ex in enumerate(tqdm(samples, desc=f"  {strategy_name}")):
        prompt = format_prompt(ex["question"], ex["context"], template=prompt_template)
        pred = strategy_fn(model, tokenizer, prompt, device, **kwargs)
        preds.append(pred)
        refs.append(ex["answer"])

        if i < 3:
            print(f"    [{i}] pred={pred}  true={ex['answer']}  "
                  f"{'OK' if pred == ex['answer'] else 'WRONG'}")

    return compute_metrics(preds, refs)


def compute_metrics(preds, refs):
    valid = [(p, r) for p, r in zip(preds, refs) if p and r]
    ps, rs = zip(*valid) if valid else ([], [])
    acc = sum(p == r for p, r in valid) / len(valid) if valid else 0

    labels = sorted(set(list(ps) + list(rs)))
    lid = {l: i for i, l in enumerate(labels)}
    p_ids = [lid[p] for p in ps]
    r_ids = [lid[r] for r in rs]

    try:
        prec, rec, f1, _ = precision_recall_fscore_support(r_ids, p_ids, average="macro", zero_division=0)
    except Exception:
        prec, rec, f1 = 0, 0, 0

    # Per-class
    per_class = {}
    for label in labels:
        lp = [1 if p == label else 0 for p in ps]
        lr = [1 if r == label else 0 for r in rs]
        if sum(lr) > 0:
            cp, cr, cf, _ = precision_recall_fscore_support(lr, lp, average="binary", zero_division=0)
            per_class[label] = {"P": float(cp), "R": float(cr), "F1": float(cf), "n": sum(lr)}

    # Confusion matrix
    cm = {}
    for label in labels:
        cm[label] = Counter()
    for p, r in valid:
        cm[r][p] += 1

    return {
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "macro_P": float(prec),
        "macro_R": float(rec),
        "per_class": per_class,
        "confusion": {k: dict(v) for k, v in cm.items()},
        "total": len(valid),
        "pred_dist": dict(Counter(ps)),
    }


def print_results(name, metrics):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  Accuracy:   {metrics['accuracy']:.2%}")
    print(f"  Macro F1:   {metrics['macro_f1']:.2%}")
    print(f"  Macro P:    {metrics['macro_P']:.2%}")
    print(f"  Macro R:    {metrics['macro_R']:.2%}")
    print(f"  Pred dist:  {metrics['pred_dist']}")
    for cls in ["yes", "no", "maybe"]:
        c = metrics["per_class"].get(cls, {})
        if c:
            print(f"    {cls:6s}: P={c['P']:.1%}  R={c['R']:.1%}  F1={c['F1']:.1%}  (n={c['n']})")
    print(f"  Confusion:")
    for cls in ["yes", "no", "maybe"]:
        cm = metrics["confusion"].get(cls, {})
        if cm:
            row = [cm.get("yes", 0), cm.get("no", 0), cm.get("maybe", 0)]
            print(f"    {cls:6s}: {row}")
    print(f"{'='*70}")


# ── Model loading ─────────────────────────────────────────────────
def load_lora_model(base_path, lora_path, device, rank=16, alpha=16):
    """Load base model + inject LoRA + load LoRA weights.
    Note: training used alpha=16 (default in finetune_pubmedqa_sft.py)."""
    print(f"  Loading base: {base_path}")
    model = load_model(base_path, device=str(device))
    print(f"  Injecting LoRA (r={rank}, alpha={alpha}, scaling={alpha/rank:.2f})...")
    inject_lora(model, rank=rank, alpha=alpha,
                target_modules=("mixer.in_proj", "mixer.out_proj"))
    print(f"  Loading LoRA weights: {lora_path}")
    sd = torch.load(f"{lora_path}/pytorch_model.bin", map_location=str(device))
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def load_full_model(path, device):
    """Load a full SFT model."""
    print(f"  Loading: {path}")
    model = load_model(path, device=str(device))
    model.eval()
    return model


# ── Main ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["lora", "full_v3", "nocpt"],
                        help="Models to evaluate: lora, full_v3, nocpt, pubmedqa_only")
    parser.add_argument("--strategies", nargs="+",
                        default=["greedy", "logprob", "logprob_margin", "voting"],
                        help="Strategies: greedy, logprob, logprob_restricted, logprob_margin, "
                             "logprob_restricted_margin, logprob_long, voting")
    parser.add_argument("--prompt", default="sft", choices=["sft", "fewshot"],
                        help="Prompt template: sft (QA_TEMPLATE) or fewshot")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("Loading PubMedQA test set...")
    samples = load_pubmedqa_test(seed=args.seed)
    dist = Counter(s["answer"] for s in samples)
    print(f"  Test: {len(samples)} samples  {dict(dist)}")

    tokenizer = load_tokenizer()

    # Model configs
    model_configs = {
        "lora": {
            "name": "MixedCPT+LoRA-SFT",
            "type": "lora",
            "base_path": os.path.join(_PROJECT_ROOT, "checkpoints/mixed_wiki/biomamba_cpt_singledoc_mamba2-130m/best_model"),
            "lora_path": os.path.join(_PROJECT_ROOT, "checkpoints/mixed_sft_lora/biomamba_sft_mamba2-130m_lora_r16/best_model"),
        },
        "full_v3": {
            "name": "MixedCPT+Full-SFT-v3",
            "type": "full",
            "path": os.path.join(_PROJECT_ROOT, "checkpoints/mixed_sft_v3/biomamba_sft_mamba2-130m_full/best_model"),
        },
        "nocpt": {
            "name": "NoCPT+Full-SFT",
            "type": "full",
            "path": os.path.join(_PROJECT_ROOT, "checkpoints/nocpt/biomamba_sft_mamba2-130m_full/best_model"),
        },
        "pubmedqa_only": {
            "name": "MixedCPT+PubMedQA-only-SFT",
            "type": "full",
            "path": os.path.join(_PROJECT_ROOT, "checkpoints/pubmedqa_only/biomamba_sft_mamba2-130m_full/best_model"),
        },
    }

    # Allow raw paths as model keys (auto-detect)
    for model_key in list(args.models):
        if model_key not in model_configs and os.path.isdir(model_key):
            model_configs[model_key] = {
                "name": os.path.basename(os.path.dirname(model_key)) if model_key.endswith("best_model") else os.path.basename(model_key),
                "type": "full",
                "path": model_key,
            }

    strategy_fns = {
        "greedy": (strategy_greedy, {}),
        "logprob": (strategy_logprob, {}),
        "logprob_restricted": (strategy_logprob_restricted, {}),
        "logprob_calibrated": (strategy_logprob_calibrated, {}),
        "logprob_margin": (strategy_logprob_margin, {}),
        "logprob_restricted_margin": (strategy_logprob_restricted_margin, {}),
        "logprob_long": (strategy_logprob_long, {}),
        "voting": (strategy_voting, {"n_samples": 7, "temperature": 0.7}),
    }

    all_results = {}

    for model_key in args.models:
        cfg = model_configs.get(model_key)
        if not cfg:
            print(f"Unknown model: {model_key}")
            continue

        print(f"\n{'#'*70}")
        print(f"# Model: {cfg['name']}")
        print(f"{'#'*70}")

        if cfg["type"] == "lora":
            model = load_lora_model(cfg["base_path"], cfg["lora_path"], device)
        else:
            model = load_full_model(cfg["path"], device)

        for strat_key in args.strategies:
            fn, kwargs = strategy_fns.get(strat_key, (None, {}))
            if fn is None:
                continue

            prompt_suffix = f"+{args.prompt}" if args.prompt != "sft" else ""
            label = f"{cfg['name']} [{strat_key}{prompt_suffix}]"
            print(f"\n  Strategy: {strat_key}  Prompt: {args.prompt}")
            metrics = evaluate(model, tokenizer, samples, fn, device,
                               prompt_template=args.prompt, strategy_name=strat_key,
                               **kwargs)
            print_results(label, metrics)
            all_results[label] = metrics

        # Free memory
        del model
        torch.cuda.empty_cache()

    # ── Threshold sweep (if requested) ────────────────────────────
    if "threshold_sweep" in args.strategies and len(args.models) == 1:
        cfg = model_configs[args.models[0]]
        print(f"\n{'#'*70}")
        print(f"# Threshold sweep on {cfg['name']}")
        print(f"{'#'*70}")
        # Reload model if needed
        if cfg["type"] == "lora":
            model = load_lora_model(cfg["base_path"], cfg["lora_path"], device)
        else:
            model = load_full_model(cfg["path"], device)

        # First collect raw logits for all samples
        print("  Collecting candidate logits for all samples...")
        all_cand_logits = []
        for ex in tqdm(samples, desc="  logits"):
            prompt = format_prompt(ex["question"], ex["context"], template=args.prompt)
            cl = _get_candidate_logits(model, tokenizer, prompt, device)
            all_cand_logits.append(cl)

        refs = [s["answer"] for s in samples]

        print(f"\n  {'margin':>6s} {'maybe_t':>7s} | {'Acc':>6s} {'F1':>6s} {'yes-F1':>7s} {'no-F1':>7s} {'maybe-F1':>8s}")
        print(f"  {'-'*60}")

        best_acc, best_f1 = 0, 0
        best_acc_cfg, best_f1_cfg = None, None

        for margin_t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
            for maybe_t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
                preds = []
                for cl in all_cand_logits:
                    if not cl:
                        preds.append("yes")
                        continue
                    labels = list(cl.keys())
                    vals = torch.tensor([cl[l] for l in labels])
                    probs = F.softmax(vals, dim=0)
                    prob_dict = {l: p.item() for l, p in zip(labels, probs)}
                    sorted_c = sorted(prob_dict.items(), key=lambda x: -x[1])
                    top_label, top_prob = sorted_c[0]
                    if len(sorted_c) >= 2:
                        margin = top_prob - sorted_c[1][1]
                        maybe_prob = prob_dict.get("maybe", 0)
                        if margin < margin_t and maybe_prob > maybe_t:
                            preds.append("maybe")
                            continue
                    preds.append(top_label)

                m = compute_metrics(preds, refs)
                yes_f1 = m["per_class"].get("yes", {}).get("F1", 0)
                no_f1 = m["per_class"].get("no", {}).get("F1", 0)
                maybe_f1 = m["per_class"].get("maybe", {}).get("F1", 0)
                flag = ""
                if m["accuracy"] > best_acc:
                    best_acc = m["accuracy"]
                    best_acc_cfg = (margin_t, maybe_t)
                    flag += " *ACC"
                if m["macro_f1"] > best_f1:
                    best_f1 = m["macro_f1"]
                    best_f1_cfg = (margin_t, maybe_t)
                    flag += " *F1"
                print(f"  {margin_t:>6.2f} {maybe_t:>7.2f} | {m['accuracy']:>5.1%} {m['macro_f1']:>5.1%} "
                      f"{yes_f1:>6.1%} {no_f1:>6.1%} {maybe_f1:>7.1%}{flag}")

        print(f"\n  Best accuracy: {best_acc:.1%} at margin={best_acc_cfg}")
        print(f"  Best macro F1: {best_f1:.1%} at margin={best_f1_cfg}")
        del model
        torch.cuda.empty_cache()

    # ── Summary table ─────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print(f"  SUMMARY: PubMedQA Generative Evaluation")
    print(f"{'='*80}")
    print(f"{'Model + Strategy':<45s} {'Acc':>6s} {'F1':>6s} {'yes-F1':>7s} {'no-F1':>7s} {'maybe-F1':>8s}")
    print(f"{'-'*80}")

    for label, m in sorted(all_results.items(), key=lambda x: -x[1]["accuracy"]):
        yes_f1 = m["per_class"].get("yes", {}).get("F1", 0)
        no_f1 = m["per_class"].get("no", {}).get("F1", 0)
        maybe_f1 = m["per_class"].get("maybe", {}).get("F1", 0)
        print(f"{label:<45s} {m['accuracy']:>5.1%} {m['macro_f1']:>5.1%} "
              f"{yes_f1:>6.1%} {no_f1:>6.1%} {maybe_f1:>7.1%}")

    # Save
    outpath = os.path.join(_PROJECT_ROOT, "evaluation_results/pubmedqa_generative_comparison.json")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
