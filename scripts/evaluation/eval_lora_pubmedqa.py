#!/usr/bin/env python3
"""Evaluate a LoRA SFT checkpoint on PubMedQA."""
# --- Project root resolution (auto-generated) ---
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')
_sys.path.insert(0, _PROJECT_ROOT)
_os.chdir(_PROJECT_ROOT)
# --- End project root resolution ---

import torch
from ft_biomamba.model import load_model, load_tokenizer, inject_lora
from ft_biomamba.data import load_pubmedqa_eval
from ft_biomamba.config import EvalConfig
from ft_biomamba.evaluate import evaluate_model_on_dataset

LORA_CKPT = _os.path.join(_PROJECT_ROOT, "checkpoints/mixed_sft_lora/biomamba_sft_mamba2-130m_lora_r16/best_model")
BASE_MODEL = _os.path.join(_PROJECT_ROOT, "checkpoints/mixed_wiki/biomamba_cpt_singledoc_mamba2-130m/best_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load base model
print("Loading base model (MixedCPT)...")
model = load_model(BASE_MODEL, device=str(device))

# Inject LoRA adapters (creates the lora_layer attributes)
print("Injecting LoRA adapters...")
inject_lora(model, rank=16, alpha=32, target_modules=("mixer.in_proj", "mixer.out_proj"))

# Load the LoRA checkpoint state dict
print(f"Loading LoRA checkpoint from {LORA_CKPT}...")
state_dict = torch.load(f"{LORA_CKPT}/pytorch_model.bin", map_location=str(device))
model.load_state_dict(state_dict, strict=True)

model.eval()
tokenizer = load_tokenizer()

# Load PubMedQA
dataset = load_pubmedqa_eval("test", seed=42)
print(f"PubMedQA test: {len(dataset)} samples")

cfg = EvalConfig(dataset_name="pubmedqa", output_dir=_os.path.join(_PROJECT_ROOT, "evaluation_results"))
metrics, preds = evaluate_model_on_dataset(
    model, tokenizer, dataset, cfg, device, tag="BioMamba-LoRA-SFT", model_type="mamba2"
)

print(f"\nLoRA SFT Results on PubMedQA:")
print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
print(f"  F1:       {metrics['f1']*100:.2f}%")
print(f"  Precision:{metrics['precision']*100:.2f}%")
print(f"  Recall:   {metrics['recall']*100:.2f}%")
