"""
BioASQ / PubMedQA evaluation with optional base-model comparison.

Metrics: Accuracy, F1 (macro), Precision, Recall, per-class breakdown.
"""

import os
import re
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .config import EvalConfig, QA_TEMPLATE, T5_PROMPT, BIOGPT_PROMPT
from .model import load_model, load_tokenizer


# ===================================================================
# Model loading (Mamba2 + HuggingFace T5/CausalLM)
# ===================================================================
def load_eval_model(model_path: str, model_type: str = "mamba2", device: str = "cuda"):
    """Load a model for evaluation. Supports mamba2, t5, and auto detection."""
    if model_type == "auto":
        model_type = _detect_model_type(model_path)

    if model_type == "mamba2":
        model = load_model(model_path, device=device)
        tokenizer = load_tokenizer()
    elif model_type == "t5":
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model.to(device)
    elif model_type == "biogpt":
        from transformers import AutoTokenizer, AutoModelForCausalLM
        try:
            from transformers import BioGptTokenizer, BioGptForCausalLM
            tokenizer = BioGptTokenizer.from_pretrained(model_path)
            model = BioGptForCausalLM.from_pretrained(model_path)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path)
        model.to(device)
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {model_type} model: {model_path}  ({total_params:,} params)")
    return model, tokenizer, model_type


def _detect_model_type(model_path: str) -> str:
    p = model_path.lower()
    if "t5" in p or "flan" in p:
        return "t5"
    if "mamba" in p or "state-spaces" in p:
        return "mamba2"
    if "biogpt" in p:
        return "biogpt"
    return "causal"


# ===================================================================
# Prompt formatting
# CRITICAL: eval prompt MUST match the SFT training template exactly,
# otherwise the model sees an OOD input and performance collapses.
# ===================================================================
def format_qa_prompt(question: str, context: str, max_ctx_chars: int = 2400) -> str:
    """Format using the SAME QA_TEMPLATE used during SFT training."""
    trunc = context[:max_ctx_chars]
    if len(context) > max_ctx_chars:
        trunc += "..."
    return QA_TEMPLATE.format(question=question, context=trunc)


FEWSHOT_YESNO_PROMPT = (
    "Answer biomedical questions with ONLY one word: yes, no, or maybe.\n\n"
    "Example 1:\nQuestion: Is aspirin used to treat headaches?\nAnswer: yes\n\n"
    "Example 2:\nQuestion: Does vitamin C cure cancer?\nAnswer: no\n\n"
    "Now answer this question:\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\nAnswer:"
)


def format_fewshot_prompt(question: str, context: str, max_ctx_chars: int = 2400) -> str:
    """Few-shot prompt for base models that can't follow instruction templates."""
    trunc = context[:max_ctx_chars]
    if len(context) > max_ctx_chars:
        trunc += "..."
    return FEWSHOT_YESNO_PROMPT.format(question=question, context=trunc)


def format_t5_prompt(question: str, context: str, max_ctx_chars: int = 800) -> str:
    """T5 instruction prompt (shorter context since T5 has 512 token limit)."""
    trunc = context[:max_ctx_chars]
    if len(context) > max_ctx_chars:
        trunc += "..."
    return T5_PROMPT.format(question=question, context=trunc)


def format_biogpt_prompt(question: str, context: str, max_ctx_chars: int = 400) -> str:
    """BioGPT prompt (short context, simple format for non-instruction-tuned LM)."""
    trunc = context[:max_ctx_chars]
    if len(context) > max_ctx_chars:
        trunc += "..."
    return BIOGPT_PROMPT.format(question=question, context=trunc)


# ===================================================================
# Answer extraction & normalisation
# ===================================================================
def extract_answer(text: str, question_type: str = "yesno") -> str:
    text = text.strip().split("\n")[0].lower()
    # Clean common prefixes
    for p in ("answer:", "the answer is", "a:", "(yes/no):", "yes/no:"):
        text = text.replace(p, "")
    text = text.lstrip(" \t:;,-.")

    if question_type in ("factoid", "list", "summary"):
        text = re.sub(r"\s+", " ", text).strip(" \t:;,-.\"'()[]{}")
        return text

    # yes/no/maybe
    for keyword in ("yes", "no", "maybe"):
        if text.startswith(keyword):
            return keyword

    words = text.split()
    for w in words[:5]:
        w = w.strip(".,;:!?()[]{}\"' ")
        if w in ("yes", "no", "maybe", "uncertain", "unclear"):
            return "maybe" if w in ("uncertain", "unclear") else w

    return words[0] if words else text


def normalize_answer(answer: str) -> str:
    a = answer.lower().strip()
    if a in ("yes", "y", "true", "1"):
        return "yes"
    if a in ("no", "n", "false", "0"):
        return "no"
    if a in ("maybe", "uncertain", "unclear", "unknown"):
        return "maybe"
    return a


# ===================================================================
# Multiple-choice prompt & evaluation (MedQA, etc.)
# ===================================================================
def format_mcq_prompt(question: str, options: dict) -> str:
    """Format a multiple-choice question prompt."""
    opts = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    return (
        f"Answer the following medical question by selecting A, B, C, or D.\n\n"
        f"Question: {question}\n\n"
        f"{opts}\n\n"
        f"Answer:"
    )


@torch.no_grad()
def _logprob_mcq(model, tokenizer, prompt: str, choices: list, device,
                 max_length: int = 512, is_seq2seq: bool = False) -> str:
    """Pick the choice with highest logprob for the first token (A/B/C/D)."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"].to(device)

    if is_seq2seq:
        # For T5: use decoder_start_token + force first token
        decoder_ids = torch.tensor([[model.config.decoder_start_token_id]], device=device)
        outputs = model(input_ids=input_ids, decoder_input_ids=decoder_ids)
        logits = outputs.logits[:, -1, :]
    else:
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]

    probs = torch.softmax(logits, dim=-1)[0]
    best_choice, best_prob = choices[0], -1.0
    for c in choices:
        for variant in [c, f" {c}", c.lower(), f" {c.lower()}"]:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if ids:
                p = probs[ids[0]].item()
                if p > best_prob:
                    best_prob = p
                    best_choice = c
    return best_choice


def evaluate_mcq(model, tokenizer, dataset, cfg: EvalConfig, device,
                 model_type: str = "mamba2", tag: str = "") -> Tuple[Dict, List]:
    """Evaluate on multiple-choice dataset (MedQA)."""
    preds, refs, results = [], [], []
    is_seq2seq = model_type == "t5"

    for idx, ex in enumerate(tqdm(dataset, desc=f"Eval {tag}")):
        question = ex["question"]
        options = ex["options"]  # {"A": "...", "B": "...", ...}
        true_answer = ex["answer_idx"]  # "A", "B", "C", or "D"

        prompt = format_mcq_prompt(question, options)
        pred = _logprob_mcq(model, tokenizer, prompt, list(options.keys()),
                            device, cfg.max_length, is_seq2seq)

        preds.append(pred)
        refs.append(true_answer)

        if cfg.save_predictions:
            results.append({
                "idx": idx, "question": question[:200],
                "true": true_answer, "pred": pred,
                "correct": pred == true_answer,
            })
        if idx < 3:
            print(f"  [{idx}] pred={pred}  true={true_answer}  {'OK' if pred==true_answer else 'WRONG'}")

    acc = sum(p == r for p, r in zip(preds, refs)) / len(preds) if preds else 0
    # For MCQ, F1/precision/recall computed as macro over classes
    labels = sorted(set(preds + refs))
    lid = {l: i for i, l in enumerate(labels)}
    try:
        from sklearn.metrics import precision_recall_fscore_support
        p_ids = [lid.get(p, 0) for p in preds]
        r_ids = [lid.get(r, 0) for r in refs]
        prec, rec, f1, _ = precision_recall_fscore_support(r_ids, p_ids, average="macro", zero_division=0)
    except Exception:
        prec, rec, f1 = 0.0, 0.0, 0.0

    metrics = {
        "accuracy": float(acc), "f1": float(f1),
        "precision": float(prec), "recall": float(rec),
        "total": len(preds), "valid": len(preds), "errors": 0,
    }
    return metrics, results


# ===================================================================
# Generation
# ===================================================================
@torch.no_grad()
def _logprob_yesno(model, tokenizer, prompt: str, device, max_length: int = 512) -> str:
    """Use logprob comparison P(yes) vs P(no) for non-instruction-tuned models.
    This avoids the generation failure where models output random tokens."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"].to(device)
    outputs = model(input_ids)
    next_logits = outputs.logits[:, -1, :]  # logits for next token
    probs = torch.softmax(next_logits, dim=-1)[0]

    # Find token IDs for yes/no/maybe candidates
    candidates = {}
    for word in ["yes", "no", "maybe", "Yes", "No", "Maybe", " yes", " no", " maybe",
                 " Yes", " No", " Maybe"]:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if ids:
            token_id = ids[0]
            base = word.strip().lower()
            if base not in candidates or probs[token_id].item() > candidates[base]:
                candidates[base] = probs[token_id].item()

    if not candidates:
        return "yes"  # fallback

    return max(candidates, key=candidates.get)


@torch.no_grad()
def generate_answer(model, tokenizer, prompt: str, device, cfg: EvalConfig,
                    question_type: str = "yesno", model_type: str = "mamba2") -> str:
    # For non-instruction-tuned causal LMs (BioGPT), use logprob comparison
    if model_type == "biogpt" and question_type in ("yesno", ""):
        return _logprob_yesno(model, tokenizer, prompt, device, cfg.max_length)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=cfg.max_length)
    input_ids = inputs["input_ids"].to(device)
    input_len = input_ids.shape[1]

    is_seq2seq = model_type == "t5"

    if is_seq2seq:
        # T5 / Seq2Seq generation
        gen_kwargs = {
            "max_new_tokens": cfg.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if cfg.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = cfg.temperature
            gen_kwargs["top_p"] = cfg.top_p
        outputs = model.generate(input_ids, **gen_kwargs)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    elif model_type == "mamba2":
        # Mamba2 generation
        gen_kwargs = {
            "max_length": input_len + cfg.max_new_tokens,
            "cg": device.type == "cuda",
            "return_dict_in_generate": False,
            "output_scores": False,
        }
        if cfg.temperature > 0:
            gen_kwargs["temperature"] = cfg.temperature
            gen_kwargs["top_p"] = cfg.top_p
        try:
            outputs = model.generate(input_ids, **gen_kwargs)
        except Exception:
            generated = input_ids
            for _ in range(cfg.max_new_tokens):
                out = model(generated)
                next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_id], dim=1)
                if tokenizer.eos_token_id and next_id.item() == tokenizer.eos_token_id:
                    break
            outputs = generated
        new_ids = outputs[0][input_len:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True) if new_ids.numel() > 0 else ""
    else:
        # Standard HuggingFace causal LM
        gen_kwargs = {
            "max_new_tokens": cfg.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if cfg.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = cfg.temperature
            gen_kwargs["top_p"] = cfg.top_p
        outputs = model.generate(input_ids, **gen_kwargs)
        new_ids = outputs[0][input_len:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True) if new_ids.numel() > 0 else ""

    return extract_answer(text, question_type)


# ===================================================================
# Core evaluation loop
# ===================================================================
def _get_context(example: dict) -> str:
    """Extract context text from either BioASQ or PubMedQA example."""
    # BioASQ
    snippets = example.get("snippets", [])
    if snippets:
        parts = []
        for s in snippets:
            t = s.get("text", s) if isinstance(s, dict) else str(s)
            if t.strip():
                parts.append(t.strip())
        return "\n\n".join(parts)
    # PubMedQA
    ctx = example.get("context", {})
    if isinstance(ctx, dict):
        return "\n\n".join(ctx.get("contexts", []))
    return str(ctx)


def evaluate_model_on_dataset(model, tokenizer, dataset, cfg: EvalConfig, device,
                              tag: str = "", use_fewshot: bool = False,
                              model_type: str = "mamba2") -> Tuple[Dict, List]:
    """Run model on dataset and return (metrics_dict, prediction_list)."""
    preds, refs, results = [], [], []
    subset = dataset
    if cfg.max_samples and cfg.max_samples < len(dataset):
        import random
        random.seed(42)
        indices = random.sample(range(len(dataset)), cfg.max_samples)
        subset = dataset.select(indices)

    # Select prompt format based on model type
    if use_fewshot:
        prompt_fn = format_fewshot_prompt
    elif model_type == "t5":
        prompt_fn = format_t5_prompt
    elif model_type == "biogpt":
        prompt_fn = format_biogpt_prompt
    else:
        prompt_fn = format_qa_prompt

    for idx, ex in enumerate(tqdm(subset, desc=f"Eval {tag}")):
        try:
            if cfg.dataset_name == "bioasq":
                question = ex.get("question", ex.get("body", ""))
                qtype = str(ex.get("question_type", ex.get("type", "yesno"))).lower()
                true_raw = ex.get("exact_answer", ex.get("answer", ""))
            else:
                question = ex["question"]
                qtype = "yesno"
                true_raw = ex.get("final_decision", ex.get("answer", ""))

            context = _get_context(ex)
            prompt = prompt_fn(question, context)

            pred = generate_answer(model, tokenizer, prompt, device, cfg, qtype, model_type=model_type)
            pred_norm = normalize_answer(pred)

            if isinstance(true_raw, list):
                true_norm = normalize_answer(str(true_raw[0])) if true_raw else ""
            else:
                true_norm = normalize_answer(str(true_raw))

            preds.append(pred_norm)
            refs.append(true_norm)

            if cfg.save_predictions:
                results.append({
                    "idx": idx, "question": question[:200],
                    "true": true_norm, "pred": pred_norm,
                    "correct": pred_norm == true_norm,
                    "type": qtype,
                })

            if idx < 3:
                print(f"  [{idx}] pred={pred_norm}  true={true_norm}  {'OK' if pred_norm==true_norm else 'WRONG'}")

        except Exception as e:
            print(f"  Error on sample {idx}: {e}")
            preds.append("error")
            refs.append("unknown")

    metrics = _calc_metrics(preds, refs)
    return metrics, results


def _calc_metrics(preds: List[str], refs: List[str]) -> Dict:
    valid = [(p, r) for p, r in zip(preds, refs) if p != "error" and r != "unknown"]
    if not valid:
        return {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0,
                "total": len(preds), "valid": 0, "errors": len(preds)}

    ps, rs = zip(*valid)
    acc = sum(p == r for p, r in valid) / len(valid)

    labels = list(set(ps + rs))
    lid = {l: i for i, l in enumerate(labels)}
    p_ids = [lid.get(p, -1) for p in ps]
    r_ids = [lid.get(r, -1) for r in rs]
    try:
        prec, rec, f1, _ = precision_recall_fscore_support(r_ids, p_ids, average="macro", zero_division=0)
    except Exception:
        prec, rec, f1 = 0.0, 0.0, 0.0

    # Per-class
    per_class = {}
    for label in labels:
        lp = [1 if p == label else 0 for p in ps]
        lr = [1 if r == label else 0 for r in rs]
        if sum(lr) > 0:
            try:
                cp, cr, cf, _ = precision_recall_fscore_support(lr, lp, average="binary", zero_division=0)
                per_class[label] = {"precision": float(cp), "recall": float(cr), "f1": float(cf), "support": sum(lr)}
            except Exception:
                pass

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "total": len(preds),
        "valid": len(valid),
        "errors": len(preds) - len(valid),
        "per_class": per_class,
    }


# ===================================================================
# Public entry: run evaluation (optionally compare with base model)
# ===================================================================
def run_evaluation(cfg: EvalConfig, dataset) -> Dict:
    """Evaluate fine-tuned model (and optionally base model) on dataset."""
    if torch.cuda.is_available():
        device = torch.device("cuda", cfg.gpu_id)
        torch.cuda.set_device(cfg.gpu_id)
    else:
        device = torch.device("cpu")

    # Load primary model
    print(f"\n[eval] Loading model: {cfg.model_path}")
    ft_model, tokenizer, detected_type = load_eval_model(
        cfg.model_path, cfg.model_type, device=str(device)
    )
    ft_metrics, ft_preds = evaluate_model_on_dataset(
        ft_model, tokenizer, dataset, cfg, device, tag="FT", model_type=detected_type
    )
    del ft_model
    torch.cuda.empty_cache()

    # Base model comparison (uses few-shot prompt since base model can't follow instructions)
    base_metrics = None
    if cfg.base_model_path:
        print(f"\n[eval] Loading base model: {cfg.base_model_path}")
        print(f"  Using few-shot prompt (base model cannot follow instruction templates)")
        base_model = load_model(cfg.base_model_path, device=str(device))
        base_model.eval()
        base_metrics, _ = evaluate_model_on_dataset(
            base_model, load_tokenizer(), dataset, cfg, device,
            tag="BASE", use_fewshot=True, model_type="mamba2"
        )
        del base_model
        torch.cuda.empty_cache()

    # Print results
    _print_results(ft_metrics, base_metrics, cfg)

    # Save
    os.makedirs(cfg.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "model_path": cfg.model_path,
        "model_type": detected_type,
        "dataset": cfg.dataset_name,
        "ft_metrics": ft_metrics,
        "base_metrics": base_metrics,
        "config": {k: str(v) for k, v in vars(cfg).items()},
    }
    out_path = os.path.join(cfg.output_dir, f"eval_{cfg.dataset_name}_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")

    if ft_preds and cfg.save_predictions:
        pred_path = os.path.join(cfg.output_dir, f"predictions_{cfg.dataset_name}_{ts}.json")
        with open(pred_path, "w") as f:
            json.dump(ft_preds, f, indent=2)

    return ft_metrics


# ===================================================================
# Multi-model benchmark
# ===================================================================
def run_benchmark(models: Dict[str, Dict], dataset, cfg: EvalConfig) -> Dict[str, Dict]:
    """Evaluate multiple models and print a comparison table.

    Args:
        models: dict of {name: {"path": str, "type": str}}
        dataset: evaluation dataset
        cfg: base EvalConfig (model_path/model_type will be overridden per model)
    Returns:
        dict of {name: metrics}
    """
    if torch.cuda.is_available():
        device = torch.device("cuda", cfg.gpu_id)
        torch.cuda.set_device(cfg.gpu_id)
    else:
        device = torch.device("cpu")

    all_results = {}
    for name, info in models.items():
        path = info["path"]
        mtype = info.get("type", "auto")
        print(f"\n{'='*65}")
        print(f"  Evaluating: {name}")
        print(f"{'='*65}")

        model, tokenizer, detected = load_eval_model(path, mtype, device=str(device))
        metrics, _ = evaluate_model_on_dataset(
            model, tokenizer, dataset, cfg, device, tag=name, model_type=detected
        )
        all_results[name] = metrics
        all_results[name]["model_path"] = path
        all_results[name]["model_type"] = detected
        all_results[name]["params"] = sum(p.numel() for p in model.parameters())

        del model
        torch.cuda.empty_cache()

    # Print comparison table
    _print_benchmark(all_results, cfg)

    # Save
    os.makedirs(cfg.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(cfg.output_dir, f"benchmark_{cfg.dataset_name}_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nBenchmark results saved to {out_path}")

    return all_results


def _print_benchmark(results: Dict[str, Dict], cfg: EvalConfig):
    print("\n" + "=" * 80)
    print(f"  BENCHMARK RESULTS — {cfg.dataset_name.upper()} ({list(results.values())[0]['valid']} samples)")
    print("=" * 80)

    # Sort by accuracy descending
    ranked = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)

    header = f"{'#':<3} {'Model':<25} {'Params':>10} {'Acc':>8} {'F1':>8} {'Prec':>8} {'Rec':>8}"
    print(f"\n{header}")
    print("-" * len(header))

    for i, (name, m) in enumerate(ranked, 1):
        params_str = f"{m['params']/1e6:.0f}M" if m.get('params') else "?"
        print(f"{i:<3} {name:<25} {params_str:>10} {m['accuracy']*100:>7.2f}% {m['f1']*100:>7.2f}% "
              f"{m['precision']*100:>7.2f}% {m['recall']*100:>7.2f}%")

    print("=" * 80)


def _print_results(ft: Dict, base: Optional[Dict], cfg: EvalConfig):
    print("\n" + "=" * 65)
    print("EVALUATION RESULTS")
    print("=" * 65)
    print(f"Dataset: {cfg.dataset_name}   Samples: {ft['valid']}/{ft['total']}")

    header = f"{'Metric':<20} {'Fine-tuned':>12}"
    if base:
        header += f" {'Base':>12} {'Delta':>10}"
    print(f"\n{header}")
    print("-" * len(header))

    for key in ("accuracy", "f1", "precision", "recall"):
        line = f"{key:<20} {ft[key]*100:>11.2f}%"
        if base:
            delta = (ft[key] - base[key]) * 100
            sign = "+" if delta >= 0 else ""
            line += f" {base[key]*100:>11.2f}% {sign}{delta:>8.2f}%"
        print(line)

    # Per-class
    if ft.get("per_class"):
        print(f"\n{'--- Per-class F1 ---':^65}")
        for cls, m in sorted(ft["per_class"].items()):
            line = f"  {cls:<16} F1={m['f1']*100:5.1f}%  P={m['precision']*100:5.1f}%  R={m['recall']*100:5.1f}%  (n={m['support']})"
            print(line)

    print("=" * 65)
