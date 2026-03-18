"""
Dataset loading and preprocessing for all stages.

- CPT:  PubMed-MEDLINE (domain-specific continue pre-training)
- SFT:  PubMedQA (+ optional BioASQ) in instruction-response format
- Eval: BioASQ / PubMedQA for QA evaluation
"""

import os
import hashlib
from typing import Optional, Any, List

from datasets import load_dataset, load_from_disk, DatasetDict, concatenate_datasets
from transformers import PreTrainedTokenizer

from .config import QA_TEMPLATE, MCQ_TEMPLATE, PROJECT_ROOT


# ===================================================================
# CPT data: PubMed-MEDLINE
# ===================================================================
def prepare_cpt_data(
    tokenizer: PreTrainedTokenizer,
    data_dir: str = os.path.join(PROJECT_ROOT, "data"),
    max_length: int = 1024,
    test_size: float = 0.05,
    num_proc: int = 8,
    seed: int = 42,
    force: bool = False,
):
    """Load, preprocess, and cache tokenized PubMed-MEDLINE for CPT."""
    cache_path = os.path.join(data_dir, f"pubmed_medline_tokenized_{max_length}")

    if os.path.exists(cache_path) and not force:
        print(f"[data] Loading cached CPT data from {cache_path}")
        ds = load_from_disk(cache_path)
        ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
        return ds

    print("[data] Downloading PubMed-MEDLINE from HuggingFace ...")
    raw = load_dataset("cyrilzakka/pubmed-medline", split="train")
    print(f"[data] Raw samples: {len(raw)}")

    # Filter empty
    raw = raw.filter(lambda x: x["content"] and len(x["content"].strip()) > 0, num_proc=num_proc)

    # Combine title + content
    def _make_text(ex):
        title = (ex.get("title") or "").strip()
        content = ex["content"].strip()
        return {"text": f"{title}\n\n{content}" if title else content}

    raw = raw.map(_make_text, num_proc=num_proc)
    raw = raw.remove_columns([c for c in raw.column_names if c != "text"])

    # Deduplicate
    print(f"[data] Before dedup: {len(raw)}")
    try:
        import pandas as pd
        from datasets import Dataset as HFDataset

        df = raw.to_pandas()
        df.drop_duplicates(subset=["text"], inplace=True)
        raw = HFDataset.from_pandas(df)
        if "__index_level_0__" in raw.column_names:
            raw = raw.remove_columns(["__index_level_0__"])
    except Exception:
        pass
    print(f"[data] After dedup: {len(raw)}")

    # Train/test split
    splits = raw.train_test_split(test_size=test_size, seed=seed)
    ds_dict = DatasetDict({"train": splits["train"], "test": splits["test"]})
    print(f"[data] Train: {len(ds_dict['train'])}, Test: {len(ds_dict['test'])}")

    # Tokenize
    def _tok(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)

    ds_dict = ds_dict.map(_tok, batched=True, num_proc=num_proc, remove_columns=["text"])
    ds_dict.set_format(type="torch", columns=["input_ids", "attention_mask"])

    os.makedirs(data_dir, exist_ok=True)
    ds_dict.save_to_disk(cache_path)
    print(f"[data] Saved to {cache_path}")
    return ds_dict


# ===================================================================
# CPT data: Enhanced (larger dataset + sequence packing)
# ===================================================================
def prepare_cpt_data_packed(
    tokenizer: PreTrainedTokenizer,
    data_dir: str = os.path.join(PROJECT_ROOT, "data"),
    max_length: int = 1024,
    max_samples: int = 2_000_000,
    test_size: float = 0.02,
    num_proc: int = 8,
    seed: int = 42,
    force: bool = False,
    dataset_name: str = "MedRAG/pubmed",
):
    """Load large PubMed dataset, tokenize, and pack sequences to eliminate padding.

    Key improvements over prepare_cpt_data():
    1. Uses MedRAG/pubmed (23.9M articles) instead of cyrilzakka/pubmed-medline (471K)
    2. Sequence packing: concatenates short abstracts into full 1024-token chunks
    3. No padding waste: each training sample is 100% real tokens
    """
    tag = f"packed_{max_length}_{max_samples // 1000}k"
    cache_path = os.path.join(data_dir, f"pubmed_cpt_{tag}")

    if os.path.exists(cache_path) and not force:
        print(f"[data] Loading cached packed CPT data from {cache_path}")
        ds = load_from_disk(cache_path)
        ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
        return ds

    print(f"[data] Loading {dataset_name} (streaming, max {max_samples:,} samples)...")
    raw_stream = load_dataset(dataset_name, split="train", streaming=True)

    # Collect texts
    texts = []
    for i, ex in enumerate(raw_stream):
        if i >= max_samples:
            break
        title = (ex.get("title") or "").strip()
        content = (ex.get("content") or "").strip()
        if not content or len(content) < 50:
            continue
        text = f"{title}\n\n{content}" if title else content
        texts.append(text)
        if (i + 1) % 500_000 == 0:
            print(f"  Loaded {i+1:,} samples ({len(texts):,} valid)...")

    print(f"[data] Collected {len(texts):,} valid abstracts")

    # Tokenize all texts without padding (just get token IDs)
    print("[data] Tokenizing (no padding)...")
    eos_id = tokenizer.eos_token_id or 0
    all_token_ids = []
    batch_size = 10000
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(batch, add_special_tokens=False, truncation=False)
        for ids in encoded["input_ids"]:
            all_token_ids.extend(ids)
            all_token_ids.append(eos_id)  # separator between documents
        if (start + batch_size) % 100000 < batch_size:
            print(f"  Tokenized {min(start + batch_size, len(texts)):,} / {len(texts):,}")

    total_tokens = len(all_token_ids)
    print(f"[data] Total tokens: {total_tokens:,}")

    # Pack into fixed-length sequences
    print(f"[data] Packing into {max_length}-token sequences...")
    packed_input_ids = []
    packed_attention_mask = []

    for start in range(0, total_tokens - max_length, max_length):
        chunk = all_token_ids[start:start + max_length]
        packed_input_ids.append(chunk)
        packed_attention_mask.append([1] * max_length)

    n_packed = len(packed_input_ids)
    print(f"[data] Packed {n_packed:,} sequences ({n_packed * max_length:,} tokens, 0% padding)")

    # Create dataset
    from datasets import Dataset as HFDataset

    ds = HFDataset.from_dict({
        "input_ids": packed_input_ids,
        "attention_mask": packed_attention_mask,
    })

    # Train/test split
    import random as _random
    _random.seed(seed)
    splits = ds.train_test_split(test_size=test_size, seed=seed)
    ds_dict = DatasetDict({"train": splits["train"], "test": splits["test"]})
    print(f"[data] Train: {len(ds_dict['train']):,}, Test: {len(ds_dict['test']):,}")

    ds_dict.set_format(type="torch", columns=["input_ids", "attention_mask"])

    os.makedirs(data_dir, exist_ok=True)
    ds_dict.save_to_disk(cache_path)
    print(f"[data] Saved to {cache_path}")
    return ds_dict


# ===================================================================
# SFT data: PubMedQA + optional BioASQ
# ===================================================================
def _normalize_yesno(raw_answer: Any) -> str:
    value = raw_answer
    if isinstance(value, list):
        value = value[0] if value else ""
        if isinstance(value, list):
            value = value[0] if value else ""
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "1"}:
        return "yes"
    if text in {"no", "n", "false", "0"}:
        return "no"
    if text in {"maybe", "unknown", "uncertain", "unclear"}:
        return "maybe"
    return ""


def _format_pubmedqa(example, template: str):
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
    answer = example["final_decision"]
    return {"instruction": instruction, "response": answer, "full_text": f"{instruction} {answer}"}


def _format_bioasq(example, template: str):
    question = (example.get("question") or example.get("body", "")).strip()
    snippets = example.get("snippets", [])
    parts = []
    for s in snippets:
        t = s.get("text", s) if isinstance(s, dict) else str(s)
        t = t.strip()
        if t:
            parts.append(t)
    context_text = "\n\n".join(parts) if parts else "N/A"
    answer = _normalize_yesno(example.get("exact_answer", example.get("answer", "")))
    instruction = template.format(question=question, context=context_text)
    full_text = f"{instruction} {answer}" if answer else instruction
    return {"instruction": instruction, "response": answer, "full_text": full_text}


def _format_medqa_sft(example, template: str):
    question = example["question"]
    options = example["options"]
    opts_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    instruction = template.format(question=question, options=opts_text)
    answer = example["answer_idx"]  # "A", "B", "C", or "D"
    return {"instruction": instruction, "response": answer, "full_text": f"{instruction} {answer}"}


def prepare_sft_data(
    tokenizer: PreTrainedTokenizer,
    data_dir: str = os.path.join(PROJECT_ROOT, "data"),
    max_length: int = 1024,
    num_proc: int = 4,
    template: str = QA_TEMPLATE,
    mix_bioasq: bool = True,
    bioasq_data_path: Optional[str] = None,
    bioasq_train_ratio: float = 0.3,
    bioasq_only: bool = False,
    mix_medqa: bool = False,
    medqa_train_ratio: float = 0.2,
    seed: int = 42,
    force: bool = False,
):
    """Load PubMedQA (+ optional BioASQ / MedQA) and return tokenized train/val splits.
    If bioasq_only=True, train on BioASQ data only (no PubMedQA).
    """
    if bioasq_only:
        h = hashlib.md5((bioasq_data_path or "").encode()).hexdigest()[:8]
        cache_key = f"sft_bioasq_only_{max_length}_{h}"
    else:
        cache_key = f"sft_pubmedqa_{max_length}"
        if mix_bioasq and bioasq_data_path:
            h = hashlib.md5(bioasq_data_path.encode()).hexdigest()[:8]
            cache_key += f"_bioasq_r{bioasq_train_ratio:.2f}_{h}"
    if mix_medqa:
        cache_key += f"_medqa_r{medqa_train_ratio:.2f}"
    cache_path = os.path.join(data_dir, cache_key)

    if os.path.exists(cache_path) and not force:
        print(f"[data] Loading cached SFT data from {cache_path}")
        ds = load_from_disk(cache_path)
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return ds

    if bioasq_only:
        # --- BioASQ only mode ---
        assert bioasq_data_path and os.path.exists(bioasq_data_path), \
            f"bioasq_only requires valid bioasq_data_path, got: {bioasq_data_path}"
        print(f"[data] Loading BioASQ only from {bioasq_data_path}")
        bioasq_raw = load_from_disk(bioasq_data_path)
        if isinstance(bioasq_raw, DatasetDict):
            bioasq_raw = bioasq_raw["train"] if "train" in bioasq_raw else list(bioasq_raw.values())[0]
        bioasq_fmt = bioasq_raw.map(lambda x: _format_bioasq(x, template), remove_columns=bioasq_raw.column_names)
        bioasq_fmt = bioasq_fmt.filter(lambda x: x["response"] in ("yes", "no", "maybe"))
        print(f"[data] BioASQ valid samples: {len(bioasq_fmt)}")
        split = bioasq_fmt.train_test_split(test_size=0.2, seed=seed)
        train_ds = split["train"]
        val_ds = split["test"]
        print(f"[data] BioASQ-only train: {len(train_ds)}, val: {len(val_ds)}")
    else:
        # --- PubMedQA ---
        print("[data] Loading PubMedQA pqa_labeled ...")
        pubmed = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        print(f"[data] PubMedQA samples: {len(pubmed)}")

        fmt_pubmed = pubmed.map(lambda x: _format_pubmedqa(x, template), remove_columns=pubmed.column_names)
        split = fmt_pubmed.train_test_split(test_size=0.2, seed=seed)
        train_ds = split["train"]
        val_ds = split["test"]

    # --- Optional BioASQ mix (only when not bioasq_only) ---
    if not bioasq_only and mix_bioasq and bioasq_data_path and os.path.exists(bioasq_data_path):
        print(f"[data] Mixing BioASQ from {bioasq_data_path}")
        bioasq_raw = load_from_disk(bioasq_data_path)
        if isinstance(bioasq_raw, DatasetDict):
            bioasq_raw = bioasq_raw["train"] if "train" in bioasq_raw else list(bioasq_raw.values())[0]

        bioasq_fmt = bioasq_raw.map(lambda x: _format_bioasq(x, template), remove_columns=bioasq_raw.column_names)
        bioasq_fmt = bioasq_fmt.filter(lambda x: x["response"] in ("yes", "no", "maybe"))
        print(f"[data] BioASQ valid samples: {len(bioasq_fmt)}")

        if len(bioasq_fmt) > 0:
            ratio = min(max(bioasq_train_ratio, 0.01), 0.99)
            target = max(1, int(len(train_ds) * ratio / (1.0 - ratio)))
            bioasq_shuf = bioasq_fmt.shuffle(seed=seed)
            if len(bioasq_shuf) >= target:
                bioasq_sampled = bioasq_shuf.select(range(target))
            else:
                reps = (target + len(bioasq_shuf) - 1) // len(bioasq_shuf)
                bioasq_sampled = concatenate_datasets([bioasq_shuf] * reps).select(range(target))
            train_ds = concatenate_datasets([train_ds, bioasq_sampled]).shuffle(seed=seed)
            print(f"[data] Mixed train: PubMedQA={len(split['train'])}, BioASQ={len(bioasq_sampled)}, Total={len(train_ds)}")

    # --- Optional MedQA MCQ mix ---
    if mix_medqa:
        print("[data] Loading MedQA-USMLE for MCQ training ...")
        medqa_raw = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
        medqa_fmt = medqa_raw.map(
            lambda x: _format_medqa_sft(x, MCQ_TEMPLATE),
            remove_columns=medqa_raw.column_names,
        )
        medqa_fmt = medqa_fmt.filter(lambda x: x["response"] in ("A", "B", "C", "D"))
        print(f"[data] MedQA valid samples: {len(medqa_fmt)}")

        if len(medqa_fmt) > 0:
            ratio = min(max(medqa_train_ratio, 0.01), 0.99)
            target = max(1, int(len(train_ds) * ratio / (1.0 - ratio)))
            medqa_shuf = medqa_fmt.shuffle(seed=seed)
            if len(medqa_shuf) >= target:
                medqa_sampled = medqa_shuf.select(range(target))
            else:
                medqa_sampled = medqa_shuf  # use all available
            prev_len = len(train_ds)
            train_ds = concatenate_datasets([train_ds, medqa_sampled]).shuffle(seed=seed)
            print(f"[data] Mixed MedQA: prev={prev_len}, MedQA={len(medqa_sampled)}, Total={len(train_ds)}")

    ds_dict = DatasetDict({"train": train_ds, "validation": val_ds})

    # Tokenize with instruction masking
    def _tokenize(examples):
        full_enc = tokenizer(
            examples["full_text"], padding="max_length", truncation=True, max_length=max_length
        )
        inst_enc = tokenizer(
            examples["instruction"], padding="max_length", truncation=True, max_length=max_length
        )
        labels = []
        for i in range(len(full_enc["input_ids"])):
            label = full_enc["input_ids"][i].copy()
            inst_len = sum(1 for t in inst_enc["input_ids"][i] if t != tokenizer.pad_token_id)
            for j in range(inst_len):
                label[j] = -100
            for j, attn in enumerate(full_enc["attention_mask"][i]):
                if attn == 0:
                    label[j] = -100
            labels.append(label)
        full_enc["labels"] = labels
        return full_enc

    ds_dict = ds_dict.map(
        _tokenize, batched=True, num_proc=min(num_proc, 4),
        remove_columns=ds_dict["train"].column_names,
    )
    # Drop samples with zero supervised tokens
    ds_dict = ds_dict.filter(lambda x: any(int(t) != -100 for t in x["labels"]))
    ds_dict.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    os.makedirs(data_dir, exist_ok=True)
    ds_dict.save_to_disk(cache_path)
    print(f"[data] Saved SFT data to {cache_path}")
    return ds_dict


def _format_bioasq_hf(example, template: str):
    """Format BioASQ from HuggingFace nanyy1025/bioasq_7b_yesno format."""
    question = example["questions"].strip()
    context_text = example["context"].strip()
    answer = example["answer"].strip().lower()
    instruction = template.format(question=question, context=context_text)
    return {"instruction": instruction, "response": answer, "full_text": f"{instruction} {answer}"}


def prepare_bioasq_sft_data(
    tokenizer: PreTrainedTokenizer,
    data_dir: str = os.path.join(PROJECT_ROOT, "data"),
    bioasq_data_path: Optional[str] = None,
    max_length: int = 1024,
    num_proc: int = 4,
    test_size: float = 0.1,
    seed: int = 42,
    force: bool = False,
):
    """Load BioASQ only and return tokenized train/val splits for dedicated BioASQ SFT.

    Uses nanyy1025/bioasq_7b_yesno (670 train) by default — no overlap with 13B test set.
    Falls back to local bioasq_data_path if provided.
    """
    cache_key = f"sft_bioasq_only_7b_{max_length}"
    cache_path = os.path.join(data_dir, cache_key)

    if os.path.exists(cache_path) and not force:
        print(f"[data] Loading cached BioASQ SFT data from {cache_path}")
        ds = load_from_disk(cache_path)
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return ds

    if bioasq_data_path and os.path.exists(bioasq_data_path):
        print(f"[data] Loading BioASQ from local path: {bioasq_data_path} ...")
        bioasq_raw = load_from_disk(bioasq_data_path)
        if isinstance(bioasq_raw, DatasetDict):
            bioasq_raw = bioasq_raw["train"] if "train" in bioasq_raw else list(bioasq_raw.values())[0]
        print(f"[data] BioASQ raw: {len(bioasq_raw)}")
        bioasq_fmt = bioasq_raw.map(
            lambda x: _format_bioasq(x, QA_TEMPLATE),
            remove_columns=bioasq_raw.column_names,
        )
    else:
        print("[data] Loading BioASQ 7b yesno from HuggingFace (nanyy1025/bioasq_7b_yesno) ...")
        bioasq_hf = load_dataset("nanyy1025/bioasq_7b_yesno", split="train")
        print(f"[data] BioASQ 7b raw: {len(bioasq_hf)}")
        bioasq_fmt = bioasq_hf.map(
            lambda x: _format_bioasq_hf(x, QA_TEMPLATE),
            remove_columns=bioasq_hf.column_names,
        )

    bioasq_fmt = bioasq_fmt.filter(lambda x: x["response"] in ("yes", "no", "maybe"))
    print(f"[data] BioASQ valid: {len(bioasq_fmt)}")

    split = bioasq_fmt.train_test_split(test_size=test_size, seed=seed)
    train_ds = split["train"]
    val_ds = split["test"]
    print(f"[data] BioASQ train: {len(train_ds)}, val: {len(val_ds)}")

    ds_dict = DatasetDict({"train": train_ds, "validation": val_ds})

    def _tokenize(examples):
        full_enc = tokenizer(
            examples["full_text"], padding="max_length", truncation=True, max_length=max_length
        )
        inst_enc = tokenizer(
            examples["instruction"], padding="max_length", truncation=True, max_length=max_length
        )
        labels = []
        for i in range(len(full_enc["input_ids"])):
            label = full_enc["input_ids"][i].copy()
            inst_len = sum(1 for t in inst_enc["input_ids"][i] if t != tokenizer.pad_token_id)
            for j in range(inst_len):
                label[j] = -100
            for j, attn in enumerate(full_enc["attention_mask"][i]):
                if attn == 0:
                    label[j] = -100
            labels.append(label)
        full_enc["labels"] = labels
        return full_enc

    ds_dict = ds_dict.map(
        _tokenize, batched=True, num_proc=min(num_proc, 4),
        remove_columns=ds_dict["train"].column_names,
    )
    ds_dict = ds_dict.filter(lambda x: any(int(t) != -100 for t in x["labels"]))
    ds_dict.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    os.makedirs(data_dir, exist_ok=True)
    ds_dict.save_to_disk(cache_path)
    print(f"[data] Saved BioASQ SFT data to {cache_path}")
    return ds_dict


def prepare_medqa_sft_data(
    tokenizer: PreTrainedTokenizer,
    data_dir: str = os.path.join(PROJECT_ROOT, "data"),
    max_length: int = 1024,
    num_proc: int = 4,
    test_size: float = 0.1,
    max_train_samples: Optional[int] = None,
    seed: int = 42,
    force: bool = False,
):
    """Load MedQA-USMLE only and return tokenized train/val splits for dedicated MCQ SFT."""
    cache_key = f"sft_medqa_only_{max_length}"
    if max_train_samples:
        cache_key += f"_n{max_train_samples}"
    cache_path = os.path.join(data_dir, cache_key)

    if os.path.exists(cache_path) and not force:
        print(f"[data] Loading cached MedQA SFT data from {cache_path}")
        ds = load_from_disk(cache_path)
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return ds

    print("[data] Loading MedQA-USMLE train split ...")
    medqa_raw = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
    print(f"[data] MedQA raw: {len(medqa_raw)}")

    medqa_fmt = medqa_raw.map(
        lambda x: _format_medqa_sft(x, MCQ_TEMPLATE),
        remove_columns=medqa_raw.column_names,
    )
    medqa_fmt = medqa_fmt.filter(lambda x: x["response"] in ("A", "B", "C", "D"))
    print(f"[data] MedQA valid: {len(medqa_fmt)}")

    if max_train_samples and max_train_samples < len(medqa_fmt):
        medqa_fmt = medqa_fmt.shuffle(seed=seed).select(range(max_train_samples))
        print(f"[data] Subsampled to {len(medqa_fmt)}")

    split = medqa_fmt.train_test_split(test_size=test_size, seed=seed)
    train_ds = split["train"]
    val_ds = split["test"]
    print(f"[data] MedQA train: {len(train_ds)}, val: {len(val_ds)}")

    ds_dict = DatasetDict({"train": train_ds, "validation": val_ds})

    def _tokenize(examples):
        full_enc = tokenizer(
            examples["full_text"], padding="max_length", truncation=True, max_length=max_length
        )
        inst_enc = tokenizer(
            examples["instruction"], padding="max_length", truncation=True, max_length=max_length
        )
        labels = []
        for i in range(len(full_enc["input_ids"])):
            label = full_enc["input_ids"][i].copy()
            inst_len = sum(1 for t in inst_enc["input_ids"][i] if t != tokenizer.pad_token_id)
            for j in range(inst_len):
                label[j] = -100
            for j, attn in enumerate(full_enc["attention_mask"][i]):
                if attn == 0:
                    label[j] = -100
            labels.append(label)
        full_enc["labels"] = labels
        return full_enc

    ds_dict = ds_dict.map(
        _tokenize, batched=True, num_proc=min(num_proc, 4),
        remove_columns=ds_dict["train"].column_names,
    )
    ds_dict = ds_dict.filter(lambda x: any(int(t) != -100 for t in x["labels"]))
    ds_dict.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    os.makedirs(data_dir, exist_ok=True)
    ds_dict.save_to_disk(cache_path)
    print(f"[data] Saved MedQA SFT data to {cache_path}")
    return ds_dict


# ===================================================================
# Eval data: BioASQ / PubMedQA
# ===================================================================
def load_bioasq_eval(data_path: Optional[str] = None, split: str = "test"):
    """Load BioASQ evaluation dataset."""
    if data_path and os.path.exists(data_path):
        ds = load_from_disk(data_path)
        if isinstance(ds, DatasetDict):
            return ds[split] if split in ds else list(ds.values())[0]
        return ds
    raise FileNotFoundError(
        "BioASQ dataset not found locally. Please provide --data_path to a processed BioASQ dataset."
    )


def load_pubmedqa_eval(split: str = "test", seed: int = 42):
    """Load PubMedQA for evaluation."""
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    splits = dataset.train_test_split(test_size=0.2, seed=seed)
    return splits["test"] if split == "test" else splits["train"]


def load_medqa_eval(split: str = "test", max_samples: Optional[int] = None):
    """Load MedQA (USMLE 4-option multiple choice) for evaluation."""
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split=split)
    if max_samples and max_samples < len(dataset):
        import random
        random.seed(42)
        indices = random.sample(range(len(dataset)), max_samples)
        dataset = dataset.select(indices)
    return dataset
