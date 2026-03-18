"""
Dataclass-based configuration for all stages of the BioMamba fine-tuning pipeline.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List

# Project root: /data/BioMamba (two levels up from this file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MAMBA2_MODELS = {
    "mamba2-130m":  {"path": "state-spaces/mamba2-130m",  "batch_size": 32},
    "mamba2-370m":  {"path": "state-spaces/mamba2-370m",  "batch_size": 16},
    "mamba2-780m":  {"path": "state-spaces/mamba2-780m",  "batch_size": 8},
    "mamba2-1.3b":  {"path": "state-spaces/mamba2-1.3b",  "batch_size": 4},
    "mamba2-2.7b":  {"path": "state-spaces/mamba2-2.7b",  "batch_size": 2},
}

TOKENIZER_NAME = "state-spaces/mamba-2.8b-hf"

# QA instruction template (shared by SFT and eval)
QA_TEMPLATE = (
    "Answer the following biomedical research question with yes, no, or maybe "
    "based on the provided context.\n\n"
    "Question: {question}\n\n"
    "Context:\n{context}\n\n"
    "Answer:"
)


# ---------------------------------------------------------------------------
# Continue Pre-Training config
# ---------------------------------------------------------------------------
@dataclass
class CPTConfig:
    # model
    model_name: str = "mamba2-130m"
    model_path: Optional[str] = None  # resume from checkpoint

    # data
    data_dir: str = os.path.join(PROJECT_ROOT, "data")
    max_length: int = 1024
    num_proc: int = 8
    test_size: float = 0.05

    # training
    batch_size: Optional[int] = None  # auto from registry
    accumulation_steps: int = 8
    lr: float = 5e-6  # conservative for continued pretraining (avoids catastrophic forgetting)
    weight_decay: float = 0.1
    num_epochs: int = 3
    warmup_ratio: float = 0.10
    stable_ratio: float = 0.65
    decay_ratio: float = 0.20
    min_lr_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # advanced
    use_ema: bool = True
    ema_decay: float = 0.999
    label_smoothing: float = 0.05
    layer_lr_decay: float = 0.9  # gentler decay to preserve lower-layer features

    # precision
    bf16: bool = True

    # logging / saving
    output_dir: str = os.path.join(PROJECT_ROOT, "checkpoints")
    log_dir: str = os.path.join(PROJECT_ROOT, "runs")
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 1000

    # hardware
    gpu_id: int = 0
    seed: int = 42

    def resolve_batch_size(self) -> int:
        if self.batch_size is not None:
            return self.batch_size
        return MAMBA2_MODELS.get(self.model_name, {}).get("batch_size", 16)

    def resolve_model_path(self) -> str:
        if self.model_path:
            return self.model_path
        return MAMBA2_MODELS[self.model_name]["path"]


# ---------------------------------------------------------------------------
# SFT config
# ---------------------------------------------------------------------------
@dataclass
class SFTConfig:
    # model
    model_name: str = "mamba2-130m"
    model_path: str = ""  # path to CPT checkpoint (required)

    # LoRA
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_targets: List[str] = field(
        default_factory=lambda: ["mixer.in_proj", "mixer.out_proj"]
    )

    # data
    data_dir: str = os.path.join(PROJECT_ROOT, "data")
    max_length: int = 1024
    num_proc: int = 4
    instruction_template: str = QA_TEMPLATE
    mix_bioasq: bool = True
    bioasq_data_path: Optional[str] = None
    bioasq_train_ratio: float = 0.3
    mix_medqa: bool = False
    medqa_train_ratio: float = 0.2

    # training
    batch_size: int = 8
    accumulation_steps: int = 4
    lr: float = 2e-5  # conservative to avoid overfitting on small QA dataset
    weight_decay: float = 0.01
    num_epochs: int = 5  # enough with early stopping on ~800 samples
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    scheduler: str = "cosine"  # cosine | linear

    # precision
    bf16: bool = True

    # logging / saving
    output_dir: str = os.path.join(PROJECT_ROOT, "checkpoints")
    log_dir: str = os.path.join(PROJECT_ROOT, "runs")
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 200

    # hardware
    gpu_id: int = 0
    seed: int = 42


# ---------------------------------------------------------------------------
# Evaluation config
# ---------------------------------------------------------------------------
T5_PROMPT = (
    "Answer this biomedical question with yes, no, or maybe.\n\n"
    "Context: {context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

BIOGPT_PROMPT = (
    "Background: {context}\n\n"
    "Question: {question}\n"
    "Answer (yes/no):"
)

MCQ_TEMPLATE = (
    "Answer the following medical question by selecting A, B, C, or D.\n\n"
    "Question: {question}\n\n"
    "{options}\n\n"
    "Answer:"
)

# Registry of external models for benchmarking
BENCHMARK_MODELS = {
    "flan-t5-small":      {"path": "google/flan-t5-small",       "type": "t5"},
    "flan-t5-base":       {"path": "google/flan-t5-base",        "type": "t5"},
    "flan-t5-large":      {"path": "google/flan-t5-large",       "type": "t5"},
    "medical-qa-t5-lora": {"path": "Adilbai/medical-qa-t5-lora", "type": "t5"},
    "biogpt":             {"path": "microsoft/biogpt",            "type": "biogpt"},
    "biogpt-large":       {"path": "microsoft/biogpt-large",      "type": "biogpt"},
}


@dataclass
class EvalConfig:
    # model
    model_path: str = ""  # path to SFT checkpoint (required)
    model_type: str = "mamba2"  # mamba2 | t5 | auto
    tokenizer_path: Optional[str] = None
    base_model_path: Optional[str] = None  # for comparison

    # data
    dataset_name: str = "bioasq"  # bioasq | pubmedqa
    data_path: Optional[str] = None
    split: str = "test"
    max_length: int = 1024
    max_samples: Optional[int] = None

    # generation
    max_new_tokens: int = 8
    temperature: float = 0.0  # greedy
    top_p: float = 0.95

    # hardware
    gpu_id: int = 0

    # output
    output_dir: str = os.path.join(PROJECT_ROOT, "evaluation_results")
    save_predictions: bool = True


# ---------------------------------------------------------------------------
# Full pipeline config
# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    cpt: CPTConfig = field(default_factory=CPTConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    skip_cpt: bool = False
    skip_sft: bool = False
    skip_eval: bool = False
