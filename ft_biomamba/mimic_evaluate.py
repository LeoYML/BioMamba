"""
MIMIC-III/IV clinical evaluation: PPL, completion, mortality, discharge generation.

Reuses model loading from ft_biomamba.model and generation from ft_biomamba.evaluate.
"""

import json
import math
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .evaluate import _logprob_yesno
from .mimic_config import MIMICEvalConfig
from .mimic_prompts import (
    format_completion_prompt,
    format_mortality_prompt,
    format_discharge_prompt,
)


# ===================================================================
# PPL evaluation (sliding window)
# ===================================================================
def evaluate_clinical_ppl(
    model,
    loader: DataLoader,
    device: str,
    has_attn_mask: bool = False,
) -> Tuple[float, float]:
    """Evaluate perplexity on tokenized clinical text chunks.

    Same approach as eval_ppl_domain.py: cross-entropy on next-token prediction.
    Returns (avg_loss, perplexity).
    """
    model.eval()
    total_loss, n_batches = 0.0, 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="  PPL eval", leave=False):
            inputs = batch["input_ids"].to(device)
            labels = inputs.clone()

            if has_attn_mask and "attention_mask" in batch:
                mask = batch["attention_mask"].to(device)
                labels[mask == 0] = -100

            with autocast("cuda", dtype=torch.bfloat16):
                out = model(inputs)
                loss = F.cross_entropy(
                    out.logits[:, :-1, :].reshape(-1, out.logits.size(-1)),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                )

            if torch.isfinite(loss):
                total_loss += loss.item()
                n_batches += 1

    avg_loss = total_loss / n_batches if n_batches > 0 else float("inf")
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return avg_loss, ppl


# ===================================================================
# Raw generation (no extract_answer post-processing)
# ===================================================================
@torch.no_grad()
def _generate_raw(model, tokenizer, prompt: str, device: torch.device,
                  max_new_tokens: int = 128, temperature: float = 0.0,
                  top_p: float = 0.95, max_length: int = 1024) -> str:
    """Generate text without yes/no extraction. Returns raw decoded output."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"].to(device)
    input_len = input_ids.shape[1]

    gen_kwargs = {
        "max_length": input_len + max_new_tokens,
        "cg": device.type == "cuda",
        "return_dict_in_generate": False,
        "output_scores": False,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    try:
        outputs = model.generate(input_ids, **gen_kwargs)
    except Exception:
        # Fallback: manual autoregressive
        generated = input_ids
        for _ in range(max_new_tokens):
            out = model(generated)
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_id], dim=1)
            if tokenizer.eos_token_id and next_id.item() == tokenizer.eos_token_id:
                break
        outputs = generated

    new_ids = outputs[0][input_len:]
    if new_ids.numel() == 0:
        return ""
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ===================================================================
# Completion evaluation (ROUGE)
# ===================================================================
def _get_rouge_scorer():
    """Lazy import rouge_score."""
    try:
        from rouge_score import rouge_scorer
        return rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    except ImportError:
        raise ImportError("pip install rouge-score  (required for completion evaluation)")


def evaluate_completion(
    model,
    tokenizer,
    samples: List[Dict],
    cfg: MIMICEvalConfig,
    device: str,
) -> Tuple[Dict, List[Dict]]:
    """Evaluate note completion: generate continuation and score with ROUGE.

    Args:
        samples: list from prepare_completion_data() with prefix_ids, reference_text.
    Returns:
        (metrics_dict, predictions_list)
    """
    scorer = _get_rouge_scorer()
    model.eval()

    all_rouge1, all_rouge2, all_rougeL = [], [], []
    predictions = []

    torch_device = torch.device(device)

    for i, sample in enumerate(tqdm(samples, desc="  Completion eval")):
        prefix_text = sample["prefix_text"]
        reference = sample["reference_text"]

        # Generate continuation (raw, no extract_answer truncation)
        try:
            prompt = format_completion_prompt(prefix_text)
            generated = _generate_raw(
                model, tokenizer, prompt, torch_device,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                max_length=cfg.max_length,
            )
        except Exception as e:
            generated = ""
            if i < 3:
                print(f"  [warn] Generation error on sample {i}: {e}")

        # Score
        if generated and reference:
            scores = scorer.score(reference, generated)
            all_rouge1.append(scores["rouge1"].fmeasure)
            all_rouge2.append(scores["rouge2"].fmeasure)
            all_rougeL.append(scores["rougeL"].fmeasure)
        else:
            all_rouge1.append(0.0)
            all_rouge2.append(0.0)
            all_rougeL.append(0.0)

        if cfg.save_predictions:
            predictions.append({
                "idx": i,
                "prefix": prefix_text[:200] + "...",
                "generated": generated[:500],
                "reference": reference[:500],
                "rouge1": all_rouge1[-1],
                "rougeL": all_rougeL[-1],
            })

        if i < 3:
            print(f"  [{i}] ROUGE-1={all_rouge1[-1]:.3f}  ROUGE-L={all_rougeL[-1]:.3f}")
            print(f"       gen: {generated[:100]}...")

    metrics = {
        "rouge1": float(np.mean(all_rouge1)) if all_rouge1 else 0.0,
        "rouge2": float(np.mean(all_rouge2)) if all_rouge2 else 0.0,
        "rougeL": float(np.mean(all_rougeL)) if all_rougeL else 0.0,
        "total": len(samples),
        "valid": sum(1 for r in all_rouge1 if r > 0),
    }
    return metrics, predictions


# ===================================================================
# Mortality prediction (logprob-based)
# ===================================================================
def evaluate_mortality(
    model,
    tokenizer,
    samples: List[Dict],
    cfg: MIMICEvalConfig,
    device: str,
) -> Tuple[Dict, List[Dict]]:
    """Evaluate mortality prediction using logprob comparison P(yes) vs P(no).

    Uses _logprob_yesno from evaluate.py.
    Returns (metrics_dict, predictions_list).
    """
    model.eval()
    preds, refs = [], []
    predictions = []
    torch_device = torch.device(device)

    for i, sample in enumerate(tqdm(samples, desc="  Mortality eval")):
        prompt = format_mortality_prompt(sample["context"], cfg.max_context_chars)
        true_label = sample["label"]  # "yes" (survived) or "no" (died)

        try:
            pred = _logprob_yesno(model, tokenizer, prompt, torch_device, cfg.max_length)
        except Exception as e:
            pred = "yes"  # fallback to majority class
            if i < 3:
                print(f"  [warn] Error on sample {i}: {e}")

        preds.append(pred)
        refs.append(true_label)

        if cfg.save_predictions:
            predictions.append({
                "idx": i,
                "subject_id": sample.get("subject_id"),
                "hadm_id": sample.get("hadm_id"),
                "true": true_label,
                "pred": pred,
                "correct": pred == true_label,
            })

        if i < 3:
            print(f"  [{i}] pred={pred}  true={true_label}  "
                  f"{'OK' if pred == true_label else 'WRONG'}")

    # Calculate metrics
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support, roc_auc_score,
    )

    acc = accuracy_score(refs, preds)

    # Binary: "no" (died) = positive class
    binary_refs = [1 if r == "no" else 0 for r in refs]
    binary_preds = [1 if p == "no" else 0 for p in preds]

    prec, rec, f1, _ = precision_recall_fscore_support(
        binary_refs, binary_preds, average="binary", zero_division=0
    )

    try:
        auroc = roc_auc_score(binary_refs, binary_preds)
    except ValueError:
        auroc = 0.0  # single class in predictions

    metrics = {
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "auroc": float(auroc),
        "total": len(preds),
        "survived": sum(1 for r in refs if r == "yes"),
        "died": sum(1 for r in refs if r == "no"),
    }
    return metrics, predictions


# ===================================================================
# Discharge summary generation (ROUGE)
# ===================================================================
def evaluate_discharge(
    model,
    tokenizer,
    samples: List[Dict],
    cfg: MIMICEvalConfig,
    device: str,
) -> Tuple[Dict, List[Dict]]:
    """Evaluate discharge summary generation with ROUGE scores.

    Args:
        samples: list from prepare_discharge_data() with context, reference.
    Returns:
        (metrics_dict, predictions_list)
    """
    scorer = _get_rouge_scorer()
    model.eval()

    all_rouge1, all_rouge2, all_rougeL = [], [], []
    predictions = []

    torch_device = torch.device(device)

    for i, sample in enumerate(tqdm(samples, desc="  Discharge eval")):
        prompt = format_discharge_prompt(sample["context"], cfg.max_context_chars)
        reference = sample["reference"]

        try:
            generated = _generate_raw(
                model, tokenizer, prompt, torch_device,
                max_new_tokens=min(cfg.max_new_tokens, 256),
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                max_length=cfg.max_length,
            )
        except Exception as e:
            generated = ""
            if i < 3:
                print(f"  [warn] Generation error on sample {i}: {e}")

        if generated and reference:
            scores = scorer.score(reference, generated)
            all_rouge1.append(scores["rouge1"].fmeasure)
            all_rouge2.append(scores["rouge2"].fmeasure)
            all_rougeL.append(scores["rougeL"].fmeasure)
        else:
            all_rouge1.append(0.0)
            all_rouge2.append(0.0)
            all_rougeL.append(0.0)

        if cfg.save_predictions:
            predictions.append({
                "idx": i,
                "context": sample["context"][:200] + "...",
                "generated": generated[:500],
                "reference": reference[:500],
                "rouge1": all_rouge1[-1],
                "rougeL": all_rougeL[-1],
            })

        if i < 3:
            print(f"  [{i}] ROUGE-1={all_rouge1[-1]:.3f}  ROUGE-L={all_rougeL[-1]:.3f}")
            print(f"       gen: {generated[:100]}...")

    metrics = {
        "rouge1": float(np.mean(all_rouge1)) if all_rouge1 else 0.0,
        "rouge2": float(np.mean(all_rouge2)) if all_rouge2 else 0.0,
        "rougeL": float(np.mean(all_rougeL)) if all_rougeL else 0.0,
        "total": len(samples),
        "valid": sum(1 for r in all_rouge1 if r > 0),
    }
    return metrics, predictions


# ===================================================================
# Result saving and printing
# ===================================================================
def save_results(
    task: str,
    metrics: Dict,
    predictions: List[Dict],
    cfg: MIMICEvalConfig,
):
    """Save evaluation results to JSON files."""
    os.makedirs(cfg.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    result = {
        "task": task,
        "model_path": cfg.model_path,
        "mimic_version": cfg.mimic_version,
        "metrics": metrics,
        "config": {k: str(v) for k, v in vars(cfg).items()},
        "timestamp": ts,
    }

    out_path = os.path.join(cfg.output_dir, f"mimic_{task}_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {out_path}")

    if predictions and cfg.save_predictions:
        pred_path = os.path.join(cfg.output_dir, f"mimic_{task}_predictions_{ts}.json")
        with open(pred_path, "w") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        print(f"Predictions saved to {pred_path}")

    return out_path


def print_ppl_results(model_name: str, loss: float, ppl: float):
    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"  Loss: {loss:.4f}   PPL: {ppl:.4f}")
    print(f"{'='*50}")


def print_generation_results(task: str, metrics: Dict):
    print(f"\n{'='*60}")
    print(f"  MIMIC {task.upper()} RESULTS")
    print(f"{'='*60}")

    if "accuracy" in metrics:
        # Classification task
        for key in ("accuracy", "f1", "precision", "recall", "auroc"):
            if key in metrics:
                print(f"  {key:<12}: {metrics[key]*100:>7.2f}%")
        print(f"  {'total':<12}: {metrics.get('total', 0)}")
        if "survived" in metrics:
            print(f"  {'survived':<12}: {metrics['survived']}")
            print(f"  {'died':<12}: {metrics['died']}")
    else:
        # Generation task (ROUGE)
        for key in ("rouge1", "rouge2", "rougeL"):
            if key in metrics:
                print(f"  {key:<12}: {metrics[key]*100:>7.2f}")
        print(f"  {'total':<12}: {metrics.get('total', 0)}")
        print(f"  {'valid':<12}: {metrics.get('valid', 0)}")

    print(f"{'='*60}")
