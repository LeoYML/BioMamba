"""
MIMIC-III / MIMIC-IV clinical note loading, preprocessing, and splitting.

MIMIC data requires PhysioNet credentialed access and must be downloaded locally.
This module handles both MIMIC-III (NOTEEVENTS.csv.gz) and MIMIC-IV-Note
(discharge.csv.gz, radiology.csv.gz) schemas.
"""

import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .mimic_config import MIMICEvalConfig


# ---------------------------------------------------------------------------
# Section parsing for clinical notes
# ---------------------------------------------------------------------------
SECTION_HEADERS = [
    r"CHIEF COMPLAINT",
    r"HISTORY OF (?:PRESENT |THE PRESENT )?ILLNESS",
    r"(?:PAST )?MEDICAL HISTORY",
    r"(?:ADMISSION |HOME )?MEDICATIONS?",
    r"ALLERGIES",
    r"SOCIAL HISTORY",
    r"FAMILY HISTORY",
    r"PHYSICAL EXAM(?:INATION)?",
    r"(?:PERTINENT )?(?:LABORATORY |LAB )?(?:RESULTS|DATA|FINDINGS)",
    r"(?:HOSPITAL )?COURSE",
    r"ASSESSMENT (?:AND|&) PLAN",
    r"DISCHARGE DIAGNOSIS",
    r"DISCHARGE (?:CONDITION|STATUS)",
    r"DISCHARGE INSTRUCTIONS",
    r"DISCHARGE MEDICATIONS?",
    r"FOLLOW(?:\s*-?\s*)UP",
    r"IMPRESSION",
    r"FINDINGS?",
    r"TECHNIQUE",
    r"INDICATION",
    r"COMPARISON",
    r"CONCLUSION",
]

_SECTION_RE = re.compile(
    r"(?:^|\n)\s*(?:" + "|".join(SECTION_HEADERS) + r")\s*[:\-]?\s*\n?",
    re.IGNORECASE,
)


def parse_note_sections(text: str) -> Dict[str, str]:
    """Parse a clinical note into sections by header regex.

    Returns dict of {header: content} preserving order.
    If no sections found, returns {"FULL_TEXT": text}.
    """
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        return {"FULL_TEXT": text.strip()}

    sections = {}
    for i, m in enumerate(matches):
        header = m.group().strip().rstrip(":-").strip().upper()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            sections[header] = content

    # Include any text before first section
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections = {"PREAMBLE": preamble, **sections}

    return sections


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_mimic_notes(
    mimic_dir: str,
    version: str = "iv",
    category: Optional[str] = None,
    min_length: int = 200,
) -> pd.DataFrame:
    """Load clinical notes from local MIMIC CSV files.

    Args:
        mimic_dir: Directory containing MIMIC data files.
        version: "iii" or "iv".
        category: For MIMIC-III, filter by CATEGORY (e.g. "Discharge summary").
                  For MIMIC-IV, select file: "discharge" or "radiology".
        min_length: Minimum note text length in characters.

    Returns:
        DataFrame with unified columns: [subject_id, hadm_id, category, text]
    """
    if version == "iii":
        return _load_mimic3(mimic_dir, category, min_length)
    elif version == "iv":
        return _load_mimic4(mimic_dir, category, min_length)
    else:
        raise ValueError(f"Unknown MIMIC version: {version}. Use 'iii' or 'iv'.")


def _load_mimic3(
    mimic_dir: str, category: Optional[str], min_length: int
) -> pd.DataFrame:
    """Load MIMIC-III NOTEEVENTS."""
    path = os.path.join(mimic_dir, "NOTEEVENTS.csv.gz")
    if not os.path.exists(path):
        path = os.path.join(mimic_dir, "NOTEEVENTS.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"NOTEEVENTS not found in {mimic_dir}. "
            "Download from https://physionet.org/content/mimiciii/1.4/"
        )

    print(f"Loading MIMIC-III notes from {path}...")
    cols = ["SUBJECT_ID", "HADM_ID", "CATEGORY", "TEXT"]
    df = pd.read_csv(path, usecols=cols, dtype={"SUBJECT_ID": int, "HADM_ID": "Int64"})

    # Filter by category
    if category:
        df = df[df["CATEGORY"].str.lower() == category.lower()]

    # Filter errors and short notes
    df = df.dropna(subset=["TEXT"])
    df = df[df["TEXT"].str.len() >= min_length]

    # Unify column names
    df = df.rename(columns={
        "SUBJECT_ID": "subject_id",
        "HADM_ID": "hadm_id",
        "CATEGORY": "category",
        "TEXT": "text",
    })

    print(f"  Loaded {len(df):,} notes (category={category or 'all'})")
    return df[["subject_id", "hadm_id", "category", "text"]].reset_index(drop=True)


def _load_mimic4(
    mimic_dir: str, category: Optional[str], min_length: int
) -> pd.DataFrame:
    """Load MIMIC-IV-Note (discharge or radiology)."""
    cat = (category or "discharge").lower()

    path = os.path.join(mimic_dir, f"{cat}.csv.gz")
    if not os.path.exists(path):
        path = os.path.join(mimic_dir, f"{cat}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{cat}.csv(.gz) not found in {mimic_dir}. "
            "Download from https://physionet.org/content/mimic-iv-note/2.2/"
        )

    print(f"Loading MIMIC-IV {cat} notes from {path}...")
    df = pd.read_csv(path)

    # Unify column names
    df = df.rename(columns={"note_type": "category"})
    if "category" not in df.columns:
        df["category"] = cat

    df = df.dropna(subset=["text"])
    df = df[df["text"].str.len() >= min_length]

    # Ensure required columns exist
    for col in ["subject_id", "hadm_id"]:
        if col not in df.columns:
            df[col] = 0

    print(f"  Loaded {len(df):,} {cat} notes")
    return df[["subject_id", "hadm_id", "category", "text"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Patient-level splitting (avoids data leakage)
# ---------------------------------------------------------------------------
def split_mimic_data(
    df: pd.DataFrame, test_ratio: float = 0.1, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split by subject_id to avoid data leakage between train/test."""
    rng = np.random.RandomState(seed)
    subjects = df["subject_id"].unique()
    rng.shuffle(subjects)

    n_test = max(1, int(len(subjects) * test_ratio))
    test_subjects = set(subjects[:n_test])

    test_mask = df["subject_id"].isin(test_subjects)
    train_df = df[~test_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)

    print(f"  Split: {len(train_df):,} train / {len(test_df):,} test "
          f"({len(subjects) - n_test} / {n_test} patients)")
    return train_df, test_df


# ---------------------------------------------------------------------------
# PPL dataset: tokenized chunks with sliding window
# ---------------------------------------------------------------------------
class TokenizedTextDataset(Dataset):
    """Dataset of pre-tokenized fixed-length sequences."""

    def __init__(self, input_ids_list: List[List[int]]):
        self.data = input_ids_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return {"input_ids": torch.tensor(self.data[i], dtype=torch.long)}


def prepare_perplexity_data(
    notes_df: pd.DataFrame,
    tokenizer,
    max_length: int = 1024,
    stride: int = 512,
    max_notes: Optional[int] = None,
) -> TokenizedTextDataset:
    """Tokenize clinical notes into fixed-length chunks using sliding window.

    Each note is tokenized, then split into overlapping windows of max_length
    tokens with the given stride. This avoids bias from only evaluating
    note openings.
    """
    texts = notes_df["text"].tolist()
    if max_notes and max_notes < len(texts):
        rng = np.random.RandomState(42)
        indices = rng.choice(len(texts), max_notes, replace=False)
        texts = [texts[i] for i in indices]

    all_chunks = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < max_length:
            # Pad short sequences
            ids = ids + [tokenizer.pad_token_id or tokenizer.eos_token_id] * (max_length - len(ids))
            all_chunks.append(ids[:max_length])
        else:
            # Sliding window
            for start in range(0, len(ids) - max_length + 1, stride):
                all_chunks.append(ids[start : start + max_length])

    print(f"  PPL dataset: {len(all_chunks)} chunks of {max_length} tokens "
          f"(stride={stride}) from {len(texts)} notes")
    return TokenizedTextDataset(all_chunks)


# ---------------------------------------------------------------------------
# Completion dataset: prefix + continuation pairs
# ---------------------------------------------------------------------------
def prepare_completion_data(
    notes_df: pd.DataFrame,
    tokenizer,
    prefix_ratio: float = 0.5,
    max_length: int = 1024,
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """Split each note into prefix + continuation for completion evaluation.

    Returns list of dicts with keys:
        - prefix_text: str (human-readable prefix)
        - prefix_ids: list[int] (tokenized prefix, truncated to fit max_length)
        - reference_text: str (continuation to compare against)
    """
    texts = notes_df["text"].tolist()
    if max_samples and max_samples < len(texts):
        rng = np.random.RandomState(42)
        indices = rng.choice(len(texts), max_samples, replace=False)
        texts = [texts[i] for i in indices]

    samples = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < 20:
            continue  # too short

        split_point = int(len(ids) * prefix_ratio)
        split_point = min(split_point, max_length)  # prefix fits in model context

        prefix_ids = ids[:split_point]
        continuation_ids = ids[split_point:]

        if len(continuation_ids) < 5:
            continue

        samples.append({
            "prefix_text": tokenizer.decode(prefix_ids, skip_special_tokens=True),
            "prefix_ids": prefix_ids,
            "reference_text": tokenizer.decode(continuation_ids, skip_special_tokens=True),
        })

    print(f"  Completion dataset: {len(samples)} samples (prefix_ratio={prefix_ratio})")
    return samples


# ---------------------------------------------------------------------------
# Mortality dataset: notes + outcome label
# ---------------------------------------------------------------------------
def load_admissions(admissions_path: str, version: str = "iv") -> pd.DataFrame:
    """Load admissions table for mortality labels."""
    if not os.path.exists(admissions_path):
        raise FileNotFoundError(
            f"Admissions file not found: {admissions_path}\n"
            "For MIMIC-IV: download hosp/admissions.csv.gz\n"
            "For MIMIC-III: download ADMISSIONS.csv.gz"
        )

    df = pd.read_csv(admissions_path)

    if version == "iii":
        df = df.rename(columns={
            "SUBJECT_ID": "subject_id",
            "HADM_ID": "hadm_id",
            "HOSPITAL_EXPIRE_FLAG": "mortality",
        })
    else:
        # MIMIC-IV: derive mortality from hospital_expire_flag or deathtime
        if "hospital_expire_flag" in df.columns:
            df = df.rename(columns={"hospital_expire_flag": "mortality"})
        elif "deathtime" in df.columns:
            df["mortality"] = df["deathtime"].notna().astype(int)
        else:
            raise ValueError("Cannot find mortality indicator in admissions table")

    return df[["subject_id", "hadm_id", "mortality"]].dropna(subset=["hadm_id"])


def prepare_mortality_data(
    notes_df: pd.DataFrame,
    admissions_path: str,
    version: str = "iv",
    max_context_chars: int = 4000,
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """Join notes with mortality labels for prediction task.

    Returns list of dicts with keys:
        - context: str (clinical note text, truncated)
        - label: str ("yes" for survived, "no" for died)
        - subject_id, hadm_id: int
    """
    admissions = load_admissions(admissions_path, version)

    # Merge on hadm_id
    merged = notes_df.merge(admissions, on=["subject_id", "hadm_id"], how="inner")
    print(f"  Mortality: {len(merged):,} notes matched with admissions "
          f"({merged['mortality'].sum():,} deaths)")

    if max_samples and max_samples < len(merged):
        merged = merged.sample(n=max_samples, random_state=42)

    samples = []
    for _, row in merged.iterrows():
        context = row["text"][:max_context_chars]
        label = "no" if row["mortality"] == 1 else "yes"  # survived = yes
        samples.append({
            "context": context,
            "label": label,
            "subject_id": int(row["subject_id"]),
            "hadm_id": int(row["hadm_id"]),
        })

    pos = sum(1 for s in samples if s["label"] == "yes")
    neg = len(samples) - pos
    print(f"  Mortality dataset: {len(samples)} samples (survived={pos}, died={neg})")
    return samples


# ---------------------------------------------------------------------------
# Discharge generation dataset: admission notes → discharge summary
# ---------------------------------------------------------------------------
def prepare_discharge_data(
    notes_df: pd.DataFrame,
    max_context_chars: int = 4000,
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """Prepare discharge summary generation samples.

    For each discharge note, extract the HOSPITAL COURSE or ASSESSMENT & PLAN
    section as the target, and use earlier sections as context.

    Returns list of dicts with keys:
        - context: str (admission info sections)
        - reference: str (discharge-specific sections)
        - subject_id, hadm_id: int
    """
    texts = notes_df["text"].tolist()
    meta = notes_df[["subject_id", "hadm_id"]].to_dict("records")

    if max_samples and max_samples < len(texts):
        rng = np.random.RandomState(42)
        indices = rng.choice(len(texts), max_samples, replace=False)
        texts = [texts[i] for i in indices]
        meta = [meta[i] for i in indices]

    # Sections that form the "context" (admission info)
    context_headers = {
        "CHIEF COMPLAINT", "HISTORY OF PRESENT ILLNESS",
        "HISTORY OF THE PRESENT ILLNESS", "PAST MEDICAL HISTORY",
        "MEDICAL HISTORY", "MEDICATIONS", "ADMISSION MEDICATIONS",
        "ALLERGIES", "SOCIAL HISTORY", "FAMILY HISTORY",
        "PHYSICAL EXAMINATION", "PHYSICAL EXAM",
        "LABORATORY RESULTS", "LAB RESULTS", "PERTINENT RESULTS",
        "PREAMBLE",
    }
    # Sections that form the "reference" (discharge output)
    target_headers = {
        "HOSPITAL COURSE", "ASSESSMENT AND PLAN", "ASSESSMENT & PLAN",
        "DISCHARGE DIAGNOSIS", "DISCHARGE MEDICATIONS",
        "DISCHARGE INSTRUCTIONS", "DISCHARGE CONDITION",
        "FOLLOW UP", "FOLLOW-UP",
    }

    samples = []
    for text, m in zip(texts, meta):
        sections = parse_note_sections(text)

        context_parts = []
        reference_parts = []
        for header, content in sections.items():
            if header in context_headers:
                context_parts.append(f"{header}:\n{content}")
            elif header in target_headers:
                reference_parts.append(f"{header}:\n{content}")

        if not context_parts or not reference_parts:
            continue

        context = "\n\n".join(context_parts)[:max_context_chars]
        reference = "\n\n".join(reference_parts)

        samples.append({
            "context": context,
            "reference": reference,
            "subject_id": m["subject_id"],
            "hadm_id": m["hadm_id"],
        })

    print(f"  Discharge generation dataset: {len(samples)} samples "
          f"(from {len(texts)} notes, {len(texts) - len(samples)} skipped)")
    return samples
