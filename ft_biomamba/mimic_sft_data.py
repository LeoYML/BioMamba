"""
Prepare MIMIC clinical SFT data in instruction-response format.

Two task types:
1. Note completion: given prefix → generate continuation
2. Discharge generation: given admission sections → generate discharge sections

Uses instruction masking: loss only on response tokens, not instruction.
"""

import os
import hashlib
from typing import Optional

import numpy as np
from datasets import Dataset as HFDataset, DatasetDict, load_from_disk

from .config import PROJECT_ROOT
from .mimic_data import (
    load_mimic_notes,
    split_mimic_data,
    parse_note_sections,
)
from .mimic_prompts import (
    COMPLETION_TEMPLATE,
    DISCHARGE_TEMPLATE,
    MORTALITY_TEMPLATE,
)


# ---------------------------------------------------------------------------
# Prompt templates for SFT (slightly different from eval — includes system instruction)
# ---------------------------------------------------------------------------
CLINICAL_COMPLETION_INSTRUCTION = (
    "Continue the following clinical note.\n\n{prefix}"
)

CLINICAL_DISCHARGE_INSTRUCTION = (
    "Based on the following clinical information, write the discharge summary "
    "including hospital course, discharge diagnosis, discharge medications, "
    "and discharge instructions.\n\n"
    "Clinical Notes:\n{context}\n\n"
    "Discharge Summary:"
)

# Sections that form the "context" (admission info)
CONTEXT_HEADERS = {
    "CHIEF COMPLAINT", "HISTORY OF PRESENT ILLNESS",
    "HISTORY OF THE PRESENT ILLNESS", "PAST MEDICAL HISTORY",
    "MEDICAL HISTORY", "MEDICATIONS", "ADMISSION MEDICATIONS",
    "ALLERGIES", "SOCIAL HISTORY", "FAMILY HISTORY",
    "PHYSICAL EXAMINATION", "PHYSICAL EXAM",
    "LABORATORY RESULTS", "LAB RESULTS", "PERTINENT RESULTS",
    "PREAMBLE",
}

# Sections that form the "response" (discharge output)
TARGET_HEADERS = {
    "HOSPITAL COURSE", "ASSESSMENT AND PLAN", "ASSESSMENT & PLAN",
    "DISCHARGE DIAGNOSIS", "DISCHARGE MEDICATIONS",
    "DISCHARGE INSTRUCTIONS", "DISCHARGE CONDITION",
    "FOLLOW UP", "FOLLOW-UP", "FOLLOWUP",
}


def _build_completion_samples(texts, tokenizer, max_context_chars=3000,
                               prefix_ratio=0.5, max_response_chars=2000):
    """Build note completion SFT samples."""
    samples = []
    for text in texts:
        # Use character-level splitting for simplicity
        split_point = int(len(text) * prefix_ratio)
        split_point = min(split_point, max_context_chars)

        prefix = text[:split_point].rstrip()
        continuation = text[split_point:split_point + max_response_chars].strip()

        if len(prefix) < 100 or len(continuation) < 50:
            continue

        instruction = CLINICAL_COMPLETION_INSTRUCTION.format(prefix=prefix)
        samples.append({
            "instruction": instruction,
            "response": continuation,
            "full_text": f"{instruction} {continuation}",
        })
    return samples


def _build_discharge_samples(texts, max_context_chars=3000, max_response_chars=2000):
    """Build discharge summary generation SFT samples."""
    samples = []
    for text in texts:
        sections = parse_note_sections(text)

        context_parts = []
        response_parts = []
        for header, content in sections.items():
            if header in CONTEXT_HEADERS:
                context_parts.append(f"{header}:\n{content}")
            elif header in TARGET_HEADERS:
                response_parts.append(f"{header}:\n{content}")

        if not context_parts or not response_parts:
            continue

        context = "\n\n".join(context_parts)[:max_context_chars]
        response = "\n\n".join(response_parts)[:max_response_chars]

        if len(context) < 100 or len(response) < 50:
            continue

        instruction = CLINICAL_DISCHARGE_INSTRUCTION.format(context=context)
        samples.append({
            "instruction": instruction,
            "response": response,
            "full_text": f"{instruction} {response}",
        })
    return samples


def prepare_mimic_sft_data(
    tokenizer,
    mimic_dir: str = os.path.join(PROJECT_ROOT, "data/mimic-iv-note/2.2/note"),
    mimic_version: str = "iv",
    data_dir: str = os.path.join(PROJECT_ROOT, "data"),
    max_length: int = 1024,
    max_train_samples: int = 5000,
    task: str = "both",  # "completion" | "discharge" | "both"
    max_context_chars: int = 3000,
    max_response_chars: int = 2000,
    test_size: float = 0.1,
    num_proc: int = 4,
    seed: int = 42,
    force: bool = False,
):
    """Prepare MIMIC clinical SFT data with instruction masking.

    Args:
        tokenizer: tokenizer instance
        mimic_dir: path to MIMIC note CSV files
        task: "completion", "discharge", or "both"
        max_train_samples: max number of training samples per task type
        max_context_chars: max chars for context/prefix
        max_response_chars: max chars for response/continuation

    Returns:
        DatasetDict with "train" and "validation" splits,
        columns: input_ids, attention_mask, labels (with instruction masking)
    """
    cache_key = f"mimic_sft_{task}_{max_length}_n{max_train_samples}"
    cache_path = os.path.join(data_dir, cache_key)

    if os.path.exists(cache_path) and not force:
        print(f"[data] Loading cached MIMIC SFT data from {cache_path}")
        ds = load_from_disk(cache_path)
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return ds

    # Load MIMIC notes
    category = "discharge" if mimic_version == "iv" else "Discharge summary"
    notes_df = load_mimic_notes(mimic_dir, version=mimic_version, category=category)
    train_df, _ = split_mimic_data(notes_df, test_ratio=test_size, seed=seed)

    # Sample subset for training
    if max_train_samples and max_train_samples < len(train_df):
        train_df = train_df.sample(n=max_train_samples * 2, random_state=seed)

    texts = train_df["text"].tolist()
    print(f"[data] Building MIMIC SFT samples from {len(texts)} notes (task={task})...")

    all_samples = []

    if task in ("completion", "both"):
        completion_samples = _build_completion_samples(
            texts, tokenizer, max_context_chars=max_context_chars,
            max_response_chars=max_response_chars,
        )
        if max_train_samples and len(completion_samples) > max_train_samples:
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(completion_samples), max_train_samples, replace=False)
            completion_samples = [completion_samples[i] for i in idx]
        print(f"  Completion samples: {len(completion_samples)}")
        all_samples.extend(completion_samples)

    if task in ("discharge", "both"):
        discharge_samples = _build_discharge_samples(
            texts, max_context_chars=max_context_chars,
            max_response_chars=max_response_chars,
        )
        if max_train_samples and len(discharge_samples) > max_train_samples:
            rng = np.random.RandomState(seed + 1)
            idx = rng.choice(len(discharge_samples), max_train_samples, replace=False)
            discharge_samples = [discharge_samples[i] for i in idx]
        print(f"  Discharge samples: {len(discharge_samples)}")
        all_samples.extend(discharge_samples)

    print(f"[data] Total MIMIC SFT samples: {len(all_samples)}")

    # Shuffle and split
    rng = np.random.RandomState(seed)
    rng.shuffle(all_samples)

    n_val = max(1, int(len(all_samples) * test_size))
    val_samples = all_samples[:n_val]
    train_samples = all_samples[n_val:]

    print(f"[data] Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Create HF datasets
    train_ds = HFDataset.from_dict({
        "instruction": [s["instruction"] for s in train_samples],
        "response": [s["response"] for s in train_samples],
        "full_text": [s["full_text"] for s in train_samples],
    })
    val_ds = HFDataset.from_dict({
        "instruction": [s["instruction"] for s in val_samples],
        "response": [s["response"] for s in val_samples],
        "full_text": [s["full_text"] for s in val_samples],
    })

    ds_dict = DatasetDict({"train": train_ds, "validation": val_ds})

    # Tokenize with instruction masking (loss only on response tokens)
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
            # Mask instruction tokens
            inst_len = sum(1 for t in inst_enc["input_ids"][i] if t != tokenizer.pad_token_id)
            for j in range(inst_len):
                label[j] = -100
            # Mask padding tokens
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

    print(f"[data] After filtering: Train={len(ds_dict['train'])}, Val={len(ds_dict['validation'])}")

    os.makedirs(data_dir, exist_ok=True)
    ds_dict.save_to_disk(cache_path)
    print(f"[data] Saved MIMIC SFT data to {cache_path}")
    return ds_dict
