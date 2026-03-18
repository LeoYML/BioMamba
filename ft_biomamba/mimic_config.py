"""
Configuration for MIMIC-III/IV clinical evaluation tasks.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from .config import PROJECT_ROOT


@dataclass
class MIMICEvalConfig:
    # model
    model_path: str = ""
    model_type: str = "mamba2"  # mamba2 | t5 | auto

    # MIMIC data (local files — requires PhysioNet credentialed access)
    mimic_version: str = "iv"  # "iii" or "iv"
    mimic_data_dir: str = os.path.join(PROJECT_ROOT, "data/mimic-iv-note/2.2/note")
    admissions_path: Optional[str] = None  # for mortality task: path to admissions.csv.gz

    # task
    task: str = "perplexity"  # perplexity | completion | mortality | discharge

    # context handling for long clinical notes
    max_length: int = 1024  # tokenizer max (Mamba2 trained with 1024)
    context_strategy: str = "sliding_window"  # sliding_window | truncate_tail | section
    chunk_stride: int = 512  # for sliding window PPL
    max_context_chars: int = 4000  # character-level pre-truncation for generation tasks

    # generation
    max_new_tokens: int = 128  # longer than QA (8); clinical text needs more
    temperature: float = 0.0  # greedy
    top_p: float = 0.95

    # evaluation
    max_samples: int = 500
    note_category: Optional[str] = None  # "Discharge summary", "Radiology", etc. (MIMIC-III)
    min_note_length: int = 200  # filter very short notes (chars)
    test_ratio: float = 0.1
    seed: int = 42

    # completion-specific
    prefix_ratio: float = 0.5  # fraction of note used as prefix

    # output
    output_dir: str = os.path.join(PROJECT_ROOT, "evaluation_results/mimic")
    save_predictions: bool = True

    # hardware
    gpu_id: int = 0
    batch_size: int = 16  # for PPL evaluation
