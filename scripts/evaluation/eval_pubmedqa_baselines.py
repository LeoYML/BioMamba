#!/usr/bin/env python3
"""Evaluate baseline (non-Mamba) models on PubMedQA using logprob strategy."""
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
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_TEMPLATE = (
    "Answer biomedical questions with ONLY one word: yes, no, or maybe.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\nAnswer:"
)


def load_pubmedqa_test(seed=42):
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    splits = ds.train_test_split(test_size=0.2, seed=seed)
    test = splits["test"]
    samples = []
    for ex in test:
        ctx_dict = ex["context"]
        contexts = ctx_dict["contexts"]
        labels = ctx_dict.get("labels", [])
        if labels:
            parts = [f"[{l}] {c}" for l, c in zip(labels, contexts)]
        else:
            parts = contexts
        context = "\n".join(parts)
        answer_map = {"yes": "yes", "no": "no", "maybe": "maybe"}
        answer = answer_map.get(ex["final_decision"], ex["final_decision"])
        samples.append({"question": ex["question"], "context": context, "answer": answer})
    return samples


def evaluate_logprob(model, tokenizer, samples, device):
    answer_tokens = {}
    for ans in ["yes", "no", "maybe"]:
        ids = tokenizer.encode(ans, add_special_tokens=False)
        answer_tokens[ans] = ids[0]

    preds, labels = [], []
    for sample in tqdm(samples, desc="logprob"):
        ctx = sample["context"][:2400]
        prompt = PROMPT_TEMPLATE.format(question=sample["question"], context=ctx)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        last_logits = outputs.logits[0, -1, :]
        probs = F.softmax(last_logits, dim=-1)
        scores = {ans: probs[tid].item() for ans, tid in answer_tokens.items()}
        pred = max(scores, key=scores.get)
        preds.append(pred)
        labels.append(sample["answer"])

    # Metrics
    correct = sum(p == l for p, l in zip(preds, labels))
    acc = correct / len(labels)
    label_set = ["yes", "no", "maybe"]
    p, r, f1, sup = precision_recall_fscore_support(labels, preds, labels=label_set, zero_division=0)
    return {
        "accuracy": acc,
        "macro_f1": float(f1.mean()),
        "per_class": {
            label_set[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]), "support": int(sup[i])}
            for i in range(len(label_set))
        },
        "predictions": preds,
        "labels": labels,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, help="HuggingFace model paths")
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}")
    samples = load_pubmedqa_test()
    print(f"Loaded {len(samples)} PubMedQA test samples")

    for model_path in args.models:
        print(f"\n{'='*70}")
        print(f"Model: {model_path}")
        print(f"{'='*70}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
            ).to(device).eval()

            results = evaluate_logprob(model, tokenizer, samples, device)
            print(f"  Accuracy: {results['accuracy']:.2%}")
            print(f"  Macro F1: {results['macro_f1']:.4f}")
            for cls in ["yes", "no", "maybe"]:
                pc = results["per_class"][cls]
                print(f"    {cls}: P={pc['precision']:.3f} R={pc['recall']:.3f} F1={pc['f1']:.3f} (n={pc['support']})")

            # Save
            name = model_path.replace("/", "_")
            out_path = os.path.join(_PROJECT_ROOT, f"evaluation_results/pubmedqa_baseline_{name}.json")
            with open(out_path, "w") as f:
                json.dump({"model": model_path, **{k: v for k, v in results.items() if k != "predictions" and k != "labels"}}, f, indent=2)
            print(f"  Saved to {out_path}")

            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
