#!/usr/bin/env python3
"""
PubMedQA classification evaluation with LoRA + classification head.

Replaces generative yes/no/maybe evaluation with a classification head
on top of the Mamba2 backbone, trained with LoRA fine-tuning.

Supports data augmentation from pqa_artificial (211K synthetic samples).

Usage:
  python eval_pubmedqa_cls.py --strategy lora --model_path state-spaces/mamba2-130m
  python eval_pubmedqa_cls.py --strategy lora --augment --n_artificial 2000 \
      --model_path ./checkpoints/mixed_wiki/biomamba_cpt_singledoc_mamba2-130m/best_model
"""

# --- Project root resolution (auto-generated) ---
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')
_sys.path.insert(0, _PROJECT_ROOT)
_os.chdir(_PROJECT_ROOT)
# --- End project root resolution ---

import argparse
import json
import math
import random
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset, concatenate_datasets
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm

from ft_biomamba.model import load_tokenizer, inject_lora
from ft_biomamba.config import QA_TEMPLATE

# ── Config ──────────────────────────────────────────────────────────
LABEL_MAP = {"yes": 0, "no": 1, "maybe": 2}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_LABELS = len(LABEL_MAP)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Data loading ────────────────────────────────────────────────────
def format_cls_example(example, template=QA_TEMPLATE):
    """Format a PubMedQA example for classification (prompt + integer label)."""
    question = example["question"]
    ctx_dict = example["context"]
    contexts = ctx_dict["contexts"]
    labels = ctx_dict.get("labels", [])

    parts = []
    for i, ctx in enumerate(contexts):
        if i < len(labels) and labels[i]:
            parts.append(f"{labels[i]}: {ctx}")
        else:
            parts.append(ctx)
    context_text = "\n\n".join(parts)
    instruction = template.format(question=question, context=context_text)

    label = LABEL_MAP.get(example["final_decision"], -1)
    return {"text": instruction, "label": label}


def tokenize_cls(examples, tokenizer, max_length=512):
    """Tokenize for classification — no instruction masking needed."""
    encodings = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors=None,
    )
    encodings["labels"] = examples["label"]
    encodings["attention_mask"] = encodings["attention_mask"]
    return encodings


def load_pubmedqa_cls(seed=42):
    """Load pqa_labeled with train/val/test split.

    Test: 200 samples (same split as existing generative eval for comparability).
    Train: 640, Val: 160 from the remaining 800.
    """
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")

    # First split: same as existing eval — 80/20
    split1 = dataset.train_test_split(test_size=0.2, seed=seed)
    test_ds = split1["test"]

    # Second split: train/val from remaining 800
    split2 = split1["train"].train_test_split(test_size=0.2, seed=seed)
    train_ds = split2["train"]
    val_ds = split2["test"]

    return train_ds, val_ds, test_ds


def load_augmented_data(n_artificial=2000, seed=42, balance=True):
    """Load and subsample pqa_artificial for data augmentation.

    pqa_artificial is ~93% yes, ~7% no, 0% maybe.
    If balance=True, oversample minority class to match.
    If balance=False, use natural distribution (more total data).
    """
    print(f"  Loading pqa_artificial (target {n_artificial})...")
    artificial = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")

    # Format all first, then subsample — ensures enough "no" samples
    formatted = artificial.map(
        lambda x: format_cls_example(x),
        remove_columns=artificial.column_names,
    )
    formatted = formatted.filter(lambda x: x["label"] >= 0)

    yes_ds = formatted.filter(lambda x: x["label"] == 0)
    no_ds = formatted.filter(lambda x: x["label"] == 1)
    print(f"  Artificial pool: {len(yes_ds)} yes, {len(no_ds)} no")

    if balance:
        # Take equal yes/no, up to n_artificial total
        n_each = min(n_artificial // 2, len(yes_ds), len(no_ds))
        formatted = concatenate_datasets([
            yes_ds.shuffle(seed=seed).select(range(n_each)),
            no_ds.shuffle(seed=seed).select(range(n_each)),
        ]).shuffle(seed=seed)
    else:
        # Take n_artificial total, natural distribution
        formatted = formatted.shuffle(seed=seed).select(range(min(n_artificial, len(formatted))))

    counts = {ID2LABEL[i]: 0 for i in range(NUM_LABELS)}
    for ex in formatted:
        counts[ID2LABEL[ex["label"]]] = counts.get(ID2LABEL[ex["label"]], 0) + 1
    dist = ", ".join(f"{k}={v}" for k, v in counts.items())
    print(f"  Artificial selected: {len(formatted)} [{dist}]")
    return formatted


def prepare_data(tokenizer, args):
    """Prepare train/val/test datasets."""
    train_raw, val_raw, test_raw = load_pubmedqa_cls(seed=args.seed)
    print(f"  pqa_labeled: train={len(train_raw)}, val={len(val_raw)}, test={len(test_raw)}")

    # Format labeled data
    fmt_train = train_raw.map(format_cls_example, remove_columns=train_raw.column_names)
    fmt_val = val_raw.map(format_cls_example, remove_columns=val_raw.column_names)
    fmt_test = test_raw.map(format_cls_example, remove_columns=test_raw.column_names)

    # Filter valid labels
    fmt_train = fmt_train.filter(lambda x: x["label"] >= 0)
    fmt_val = fmt_val.filter(lambda x: x["label"] >= 0)
    fmt_test = fmt_test.filter(lambda x: x["label"] >= 0)

    # Data augmentation
    if args.augment:
        aug_data = load_augmented_data(args.n_artificial, args.seed,
                                        balance=not args.no_balance_artificial)
        parts = [aug_data]
        for _ in range(args.labeled_upsample):
            parts.append(fmt_train)
        fmt_train = concatenate_datasets(parts).shuffle(seed=args.seed)
        print(f"  Augmented train: {len(fmt_train)} samples (labeled upsampled {args.labeled_upsample}x)")

    # Print label distribution
    for name, ds in [("train", fmt_train), ("val", fmt_val), ("test", fmt_test)]:
        counts = {ID2LABEL[i]: 0 for i in range(NUM_LABELS)}
        for ex in ds:
            lbl = ID2LABEL.get(ex["label"], "?")
            counts[lbl] = counts.get(lbl, 0) + 1
        dist = ", ".join(f"{k}={v}" for k, v in counts.items())
        print(f"  {name}: {len(ds)} [{dist}]")

    # Tokenize
    def tok_fn(examples):
        return tokenize_cls(examples, tokenizer, args.max_length)

    train_tok = fmt_train.map(tok_fn, batched=True, remove_columns=["text", "label"])
    val_tok = fmt_val.map(tok_fn, batched=True, remove_columns=["text", "label"])
    test_tok = fmt_test.map(tok_fn, batched=True, remove_columns=["text", "label"])

    return train_tok, val_tok, test_tok


# ── Model ───────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification."""

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # class weight tensor
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


class MambaClassificationModel(nn.Module):
    """Mamba2 backbone + pooling + MLP classification head."""

    def __init__(self, mamba_model, num_labels=3, hidden_size=768,
                 num_cat_layers=4, head_hidden=256, dropout=0.3,
                 pooling="mean"):
        super().__init__()
        self.backbone = mamba_model
        self.num_cat_layers = num_cat_layers
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.pooling = pooling

        feat_dim = hidden_size * num_cat_layers
        self.dropout = nn.Dropout(dropout)

        if pooling == "attn":
            self.attn_weight = nn.Linear(feat_dim, 1)

        if head_hidden > 0:
            self.head = nn.Sequential(
                nn.Linear(feat_dim, head_hidden),
                nn.GELU(),
                nn.LayerNorm(head_hidden),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, num_labels),
            )
        else:
            self.head = nn.Linear(feat_dim, num_labels)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        mixer = self.backbone.backbone
        n_total = len(mixer.layers)

        hidden = mixer.embedding(input_ids)
        residual = None

        layer_outputs = []
        target_start = n_total - self.num_cat_layers

        for i, layer in enumerate(mixer.layers):
            hidden, residual = layer(hidden, residual)
            if i >= target_start:
                if residual is not None:
                    layer_outputs.append(hidden + residual)
                else:
                    layer_outputs.append(hidden)

        # Apply final norm to last layer
        if hasattr(mixer, "norm_f") and mixer.norm_f is not None:
            final = mixer.norm_f(hidden + residual if residual is not None else hidden)
            if layer_outputs:
                layer_outputs[-1] = final

        if self.num_cat_layers == 1:
            feat = layer_outputs[0]
        else:
            feat = torch.cat(layer_outputs, dim=-1)
        # feat: (batch, seq_len, feat_dim)

        # Pooling: sequence-level representation
        if self.pooling == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (feat * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                pooled = feat.mean(dim=1)
        elif self.pooling == "last":
            if attention_mask is not None:
                last_idx = attention_mask.sum(dim=1).long() - 1
                pooled = feat[torch.arange(feat.size(0), device=feat.device), last_idx]
            else:
                pooled = feat[:, -1, :]
        elif self.pooling == "max":
            if attention_mask is not None:
                feat = feat.masked_fill(~attention_mask.unsqueeze(-1).bool(), -1e9)
            pooled = feat.max(dim=1).values
        elif self.pooling == "attn":
            weights = self.attn_weight(feat).squeeze(-1)
            if attention_mask is not None:
                weights = weights.masked_fill(~attention_mask.bool(), -1e9)
            weights = torch.softmax(weights, dim=1)
            pooled = (feat * weights.unsqueeze(-1)).sum(dim=1)
        else:
            pooled = feat.mean(dim=1)

        logits = self.head(self.dropout(pooled))

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return loss, logits


# ── Training strategies ─────────────────────────────────────────────
def setup_frozen(model, head_lr=1e-3):
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True
    for p in model.dropout.parameters():
        p.requires_grad = True
    if hasattr(model, "attn_weight"):
        for p in model.attn_weight.parameters():
            p.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    return [{"params": trainable, "lr": head_lr, "weight_decay": 0.0}]


def setup_lora(model, rank=16, alpha=32, lora_lr=1e-4, head_lr=5e-4):
    for p in model.backbone.parameters():
        p.requires_grad = False

    lora_params = inject_lora(model.backbone, rank=rank, alpha=alpha)

    head_params = list(model.head.parameters()) + list(model.dropout.parameters())
    if hasattr(model, "attn_weight"):
        head_params += list(model.attn_weight.parameters())
    for p in head_params:
        p.requires_grad = True

    return [
        {"params": lora_params, "lr": lora_lr, "weight_decay": 0.01},
        {"params": head_params, "lr": head_lr, "weight_decay": 0.0},
    ]


def setup_full(model, backbone_lr=1e-5, head_lr=5e-4, weight_decay=0.01):
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        p.requires_grad = True
        if "head." in name or "dropout" in name or "attn_weight" in name:
            head_params.append(p)
        else:
            backbone_params.append(p)
    return [
        {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": head_params, "lr": head_lr, "weight_decay": 0.0},
    ]


# ── Training ────────────────────────────────────────────────────────
def compute_class_weights(dataset, num_labels=NUM_LABELS):
    """Compute inverse-frequency weights for imbalanced classes."""
    counts = [0] * num_labels
    for ex in dataset:
        lbl = ex["labels"]
        if 0 <= lbl < num_labels:
            counts[lbl] += 1
    total = sum(counts)
    weights = [total / (num_labels * c) if c > 0 else 1.0 for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


def evaluate_cls(model, dataset, amp_dtype, args):
    """Evaluate classification metrics."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for start in range(0, len(dataset), args.batch_size):
            batch = dataset.select(range(start, min(start + args.batch_size, len(dataset))))
            input_ids = torch.tensor(batch["input_ids"]).to(DEVICE)
            attention_mask = torch.tensor(batch["attention_mask"]).to(DEVICE)
            labels = batch["labels"]

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=args.bf16):
                _, logits = model(input_ids, attention_mask=attention_mask)

            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels)

    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        labels=list(range(NUM_LABELS)),
        target_names=list(LABEL_MAP.keys()),
        output_dict=True, zero_division=0,
    )
    return acc, report


def train_and_eval(model, train_ds, val_ds, test_ds, param_groups, args):
    """Train classification model and evaluate on test set."""
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {total_params:,} total, {trainable_params:,} trainable ({trainable_params/total_params*100:.2f}%)")

    steps_per_epoch = math.ceil(len(train_ds) / (args.batch_size * args.accum_steps))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))

    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    amp_dtype = torch.bfloat16 if args.bf16 else None
    trainable_list = [p for p in model.parameters() if p.requires_grad]

    best_val_f1 = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        n_batches = 0
        step_in_accum = 0

        indices = list(range(len(train_ds)))
        random.shuffle(indices)

        for start in range(0, len(indices), args.batch_size):
            batch_idx = indices[start:start + args.batch_size]
            batch = train_ds.select(batch_idx)

            input_ids = torch.tensor(batch["input_ids"]).to(DEVICE)
            attention_mask = torch.tensor(batch["attention_mask"]).to(DEVICE)
            labels = torch.tensor(batch["labels"]).to(DEVICE)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=args.bf16):
                loss, _ = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = loss / args.accum_steps

            loss.backward()
            step_in_accum += 1

            if step_in_accum == args.accum_steps or (start + args.batch_size) >= len(indices):
                torch.nn.utils.clip_grad_norm_(trainable_list, max_norm=args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step_in_accum = 0

            total_loss += loss.item() * args.accum_steps
            n_batches += 1

        avg_loss = total_loss / n_batches

        # Validation
        val_acc, val_report = evaluate_cls(model, val_ds, amp_dtype, args)
        val_f1 = val_report["macro avg"]["f1-score"]
        print(f"  Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f}  val_acc={val_acc:.4f}  val_macro_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(DEVICE)
        print(f"  Loaded best model (val_macro_f1={best_val_f1:.4f})")

    # Final test evaluation
    test_acc, test_report = evaluate_cls(model, test_ds, amp_dtype, args)
    test_labels, test_preds = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(test_ds), args.batch_size):
            batch = test_ds.select(range(start, min(start + args.batch_size, len(test_ds))))
            input_ids = torch.tensor(batch["input_ids"]).to(DEVICE)
            attn_mask = torch.tensor(batch["attention_mask"]).to(DEVICE)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=args.bf16):
                _, logits = model(input_ids, attention_mask=attn_mask)
            preds = logits.argmax(dim=-1).cpu().tolist()
            test_preds.extend(preds)
            test_labels.extend(batch["labels"])

    cm = confusion_matrix(test_labels, test_preds, labels=list(range(NUM_LABELS)))

    return test_acc, test_report, cm, best_val_f1


# ── Main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="PubMedQA classification with LoRA + classification head")

    # Model
    parser.add_argument("--model_path", default="state-spaces/mamba2-130m")
    parser.add_argument("--model_name", default=None)

    # Strategy
    parser.add_argument("--strategy", choices=["frozen", "lora", "full"], default="lora")

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_lr", type=float, default=1e-4)
    parser.add_argument("--head_lr", type=float, default=5e-4)
    parser.add_argument("--backbone_lr", type=float, default=1e-5)

    # Architecture
    parser.add_argument("--pooling", choices=["mean", "last", "max", "attn"], default="mean")
    parser.add_argument("--num_cat_layers", type=int, default=4)
    parser.add_argument("--head_hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)

    # Augmentation
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--n_artificial", type=int, default=2000)
    parser.add_argument("--labeled_upsample", type=int, default=1)
    parser.add_argument("--no_balance_artificial", action="store_true",
                        help="Don't balance yes/no in artificial data")

    # Loss
    parser.add_argument("--loss", choices=["ce", "weighted_ce", "focal"], default="weighted_ce")
    parser.add_argument("--focal_gamma", type=float, default=2.0)

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5)

    # Precision
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no_bf16", dest="bf16", action="store_false")
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--output_dir", default="./evaluation_results")

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_name = args.model_name or os.path.basename(args.model_path.rstrip("/"))

    # Load data
    print("Loading PubMedQA data...")
    tokenizer = load_tokenizer()
    train_tok, val_tok, test_tok = prepare_data(tokenizer, args)

    # Load model
    print(f"\n{'='*70}")
    print(f"  Model: {model_name}")
    print(f"  Strategy: {args.strategy}")
    print(f"  Pooling: {args.pooling}")
    print(f"  Loss: {args.loss}")
    aug_str = f"aug={args.n_artificial}, upsample={args.labeled_upsample}" if args.augment else "none"
    print(f"  Augmentation: {aug_str}")
    print(f"{'='*70}")

    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    backbone = MambaLMHeadModel.from_pretrained(args.model_path, device="cpu", dtype=torch.float32)

    hidden_size = backbone.backbone.layers[0].mixer.d_model if hasattr(backbone.backbone.layers[0].mixer, "d_model") else 768
    n_layers = len(backbone.backbone.layers)
    print(f"  Hidden size: {hidden_size}, Layers: {n_layers}")

    # Adjust for frozen strategy
    if args.strategy == "frozen":
        n_cat = 1
        h_hidden = 0
        drp = 0.1
    else:
        n_cat = args.num_cat_layers
        h_hidden = args.head_hidden
        drp = args.dropout

    model = MambaClassificationModel(
        backbone, NUM_LABELS, hidden_size,
        num_cat_layers=n_cat, head_hidden=h_hidden, dropout=drp,
        pooling=args.pooling,
    )

    # Set up loss function
    if args.loss == "weighted_ce":
        weights = compute_class_weights(train_tok).to(DEVICE)
        model.loss_fn = nn.CrossEntropyLoss(weight=weights)
        print(f"  Class weights: {weights.tolist()}")
    elif args.loss == "focal":
        weights = compute_class_weights(train_tok).to(DEVICE)
        model.loss_fn = FocalLoss(alpha=weights, gamma=args.focal_gamma)
        print(f"  Focal loss: gamma={args.focal_gamma}, weights={weights.tolist()}")
    else:
        model.loss_fn = nn.CrossEntropyLoss()

    # Setup strategy
    if args.strategy == "frozen":
        param_groups = setup_frozen(model, head_lr=args.head_lr)
    elif args.strategy == "lora":
        param_groups = setup_lora(model, args.lora_rank, args.lora_alpha,
                                  args.lora_lr, args.head_lr)
    elif args.strategy == "full":
        param_groups = setup_full(model, args.backbone_lr, args.head_lr)

    # Train and evaluate
    test_acc, test_report, cm, best_val_f1 = train_and_eval(
        model, train_tok, val_tok, test_tok, param_groups, args
    )

    # Print results
    macro = test_report["macro avg"]
    print(f"\n{'='*70}")
    print(f"  {model_name} [{args.strategy}] — PubMedQA Classification")
    print(f"  Accuracy:   {test_acc*100:.2f}%")
    print(f"  Macro F1:   {macro['f1-score']*100:.2f}%")
    print(f"  Macro P:    {macro['precision']*100:.2f}%")
    print(f"  Macro R:    {macro['recall']*100:.2f}%")
    print(f"  Per-class:")
    for cls_name in LABEL_MAP:
        r = test_report.get(cls_name, {})
        sup = r.get("support", 0)
        print(f"    {cls_name:>6}: P={r.get('precision',0)*100:5.1f}%  R={r.get('recall',0)*100:5.1f}%  F1={r.get('f1-score',0)*100:5.1f}%  (n={int(sup)})")
    print(f"  Confusion matrix (yes/no/maybe):")
    for i, row in enumerate(cm):
        print(f"    {ID2LABEL[i]:>6}: {row.tolist()}")
    print(f"{'='*70}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    result = {
        "model": model_name,
        "model_path": args.model_path,
        "strategy": args.strategy,
        "pooling": args.pooling,
        "loss": args.loss,
        "augmentation": {
            "enabled": args.augment,
            "n_artificial": args.n_artificial if args.augment else 0,
            "labeled_upsample": args.labeled_upsample if args.augment else 0,
        },
        "test_accuracy": test_acc,
        "test_macro_f1": macro["f1-score"],
        "test_macro_precision": macro["precision"],
        "test_macro_recall": macro["recall"],
        "per_class": {k: test_report[k] for k in LABEL_MAP if k in test_report},
        "confusion_matrix": cm.tolist(),
        "training": {
            "best_val_f1": best_val_f1,
            "train_samples": len(train_tok),
            "val_samples": len(val_tok),
            "test_samples": len(test_tok),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        },
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "accum_steps": args.accum_steps,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_lr": args.lora_lr,
            "head_lr": args.head_lr,
            "dropout": args.dropout,
            "head_hidden": args.head_hidden,
            "num_cat_layers": args.num_cat_layers,
            "max_length": args.max_length,
        },
    }

    aug_tag = f"_aug{args.n_artificial}" if args.augment else ""
    out_path = os.path.join(args.output_dir, f"pubmedqa_cls_{model_name}_{args.strategy}{aug_tag}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")

    del model, backbone
    torch.cuda.empty_cache()

    return result


if __name__ == "__main__":
    main()
