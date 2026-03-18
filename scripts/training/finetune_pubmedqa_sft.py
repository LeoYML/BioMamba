"""
Mamba2 Supervised Fine-Tuning (SFT) with LoRA on PubMedQA Dataset
Dataset: https://huggingface.co/datasets/qiaojin/PubMedQA (pqa_labeled subset)
"""

# --- Project root resolution (auto-generated) ---
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')
_sys.path.insert(0, _PROJECT_ROOT)
_os.chdir(_PROJECT_ROOT)
# --- End project root resolution ---

import os
import sys
import torch
import math
import random
import hashlib
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from datasets import load_dataset, load_from_disk, DatasetDict, concatenate_datasets
from mamba_ssm import MambaLMHeadModel
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Any
import json

try:
    from torch.amp import autocast as torch_autocast, GradScaler as TorchGradScaler
    USE_TORCH_AMP = True
except ImportError:
    from torch.cuda.amp import autocast as torch_autocast, GradScaler as TorchGradScaler
    USE_TORCH_AMP = False

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")

# LoRA implementation (simplified for Mamba2)
class LoRALayer(torch.nn.Module):
    """Low-Rank Adaptation Layer"""
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices
        self.lora_A = torch.nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = torch.nn.Parameter(torch.zeros(rank, out_features))
        
        # Initialize
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        return (x @ self.lora_A @ self.lora_B) * self.scaling


def inject_lora_to_mamba(model, rank=8, alpha=16, target_modules=['mixer.in_proj', 'mixer.out_proj']):
    """Inject LoRA adapters into Mamba2 model"""
    lora_params = []

    def make_lora_forward(bound_original_forward):
        def new_forward(self, x):
            result = bound_original_forward(x)
            if hasattr(self, 'lora_layer'):
                result = result + self.lora_layer(x)
            return result
        return new_forward
    
    for name, module in model.named_modules():
        # Check if this module should have LoRA
        should_add_lora = any(target in name for target in target_modules)
        
        if should_add_lora and isinstance(module, torch.nn.Linear):
            # Create LoRA layer
            lora = LoRALayer(
                in_features=module.in_features,
                out_features=module.out_features,
                rank=rank,
                alpha=alpha
            )
            # Keep LoRA weights on the same device/dtype as the target linear layer.
            lora = lora.to(device=module.weight.device, dtype=module.weight.dtype)
            
            # Register as a buffer to the module
            setattr(module, 'lora_layer', lora)
            
            # Freeze original weights
            module.weight.requires_grad = False
            if module.bias is not None:
                module.bias.requires_grad = False
            
            # Track LoRA parameters
            lora_params.extend([lora.lora_A, lora.lora_B])
            
            # Monkey patch forward method
            original_forward = module.forward
            module.forward = make_lora_forward(original_forward).__get__(module, torch.nn.Linear)
    
    return lora_params


def parse_args():
    parser = argparse.ArgumentParser(description='Mamba2 SFT with LoRA on PubMedQA')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='mamba2-130m', 
                        choices=['mamba2-130m', 'mamba2-370m', 'mamba2-780m', 'mamba2-1.3b', 'mamba2-2.7b'],
                        help='Mamba2 model to use')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to pretrained/CPT model checkpoint')
    
    # LoRA arguments
    parser.add_argument('--use_lora', action='store_true', default=False,
                        help='Use LoRA for parameter-efficient fine-tuning')
    parser.add_argument('--lora_rank', type=int, default=8,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='LoRA alpha (scaling factor)')
    parser.add_argument('--lora_target_modules', type=str, nargs='+', 
                        default=['mixer.in_proj', 'mixer.out_proj'],
                        help='Target modules for LoRA')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory to save processed data')
    parser.add_argument('--max_length', type=int, default=1024,
                        help='Maximum sequence length')
    parser.add_argument('--num_proc', type=int, default=1,
                        help='Number of processes for data preprocessing')
    parser.add_argument('--instruction_template', type=str, 
                        default="Answer the following biomedical research question with yes, no, or maybe based on the provided context.\n\nQuestion: {question}\n\nContext:\n{context}\n\nAnswer:",
                        help='Instruction template for QA formatting')
    parser.add_argument('--bioasq_only', action='store_true',
                        help='Train on BioASQ data only (no PubMedQA)')
    parser.add_argument('--mix_bioasq', action='store_true',
                        help='Mix BioASQ yes/no data into PubMedQA SFT train set')
    parser.add_argument('--bioasq_data_path', type=str, default=None,
                        help='Path to local BioASQ HF dataset for SFT mixing (must contain yes/no labels)')
    parser.add_argument('--bioasq_split', type=str, default='train',
                        help='BioASQ split name used for training mix (default: train)')
    parser.add_argument('--bioasq_train_ratio', type=float, default=0.7,
                        help='Target BioASQ ratio in mixed train set, e.g. 0.7 => 70%% BioASQ, 30%% PubMedQA')
    parser.add_argument('--bioasq_max_train_samples', type=int, default=0,
                        help='Cap BioASQ train samples in mixed set (0 means auto by ratio)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size per device')
    parser.add_argument('--accumulation_steps', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup ratio')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['linear', 'cosine'],
                        help='Learning rate scheduler')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping')
    
    # Logging and saving
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--log_dir', type=str, default='./runs',
                        help='Tensorboard log directory')
    parser.add_argument('--logging_steps', type=int, default=50,
                        help='Log every N steps')
    parser.add_argument('--eval_steps', type=int, default=200,
                        help='Evaluate every N steps')
    parser.add_argument('--save_steps', type=int, default=500,
                        help='Save checkpoint every N steps')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='Use mixed precision training')
    parser.add_argument('--bf16', action='store_true', default=False,
                        help='Use bfloat16 mixed precision training (recommended on H100/A100)')
    
    # Data loading
    parser.add_argument('--reprocess_data', action='store_true',
                        help='Force reprocessing of data')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Wandb arguments
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='biomamba2-sft',
                        help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Wandb entity (username or team)')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Wandb run name (default: auto-generated)')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_pubmedqa_example(example, instruction_template):
    """Format a PubMedQA example into instruction-response format"""
    # Extract question
    question = example['question']
    
    # Extract and format context from the structured data
    context_dict = example['context']
    contexts = context_dict['contexts']
    labels = context_dict.get('labels', [])
    
    # Combine contexts with labels if available
    context_parts = []
    for i, ctx in enumerate(contexts):
        if i < len(labels) and labels[i]:
            context_parts.append(f"{labels[i]}: {ctx}")
        else:
            context_parts.append(ctx)
    
    context_text = "\n\n".join(context_parts)
    
    # Format instruction
    instruction = instruction_template.format(
        question=question,
        context=context_text
    )
    
    # Get answer
    answer = example['final_decision']
    
    # Create full text for training
    full_text = f"{instruction} {answer}"
    
    return {
        'instruction': instruction,
        'response': answer,
        'full_text': full_text
    }


def normalize_yesno_answer(raw_answer: Any) -> str:
    """Normalize various yes/no answer encodings to yes|no|maybe."""
    value = raw_answer

    if isinstance(value, list):
        if not value:
            return ""
        value = value[0]
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


def format_bioasq_example(example, instruction_template):
    """Format a BioASQ example into instruction-response format."""
    question = example.get('question', example.get('body', '')).strip()

    snippets = example.get('snippets', [])
    context_parts = []
    for snippet in snippets:
        if isinstance(snippet, dict):
            text = snippet.get('text', '')
        else:
            text = str(snippet)
        text = text.strip()
        if text:
            context_parts.append(text)

    context_text = "\n\n".join(context_parts) if context_parts else "N/A"
    answer = normalize_yesno_answer(example.get('exact_answer', example.get('answer', '')))

    instruction = instruction_template.format(
        question=question,
        context=context_text
    )
    full_text = f"{instruction} {answer}" if answer else instruction

    return {
        'instruction': instruction,
        'response': answer,
        'full_text': full_text
    }


def prepare_pubmedqa_data(args, tokenizer):
    """Load and preprocess the SFT dataset (PubMedQA, optionally mixed with BioASQ)."""
    cache_key = f"pubmedqa_pqa_labeled_tokenized_{args.max_length}"
    if getattr(args, 'bioasq_only', False):
        bioasq_path = args.bioasq_data_path or ""
        bioasq_hash = hashlib.md5(bioasq_path.encode('utf-8')).hexdigest()[:8] if bioasq_path else "none"
        cache_key = f"bioasq_only_tokenized_{args.max_length}_{bioasq_hash}"
    elif args.mix_bioasq:
        bioasq_path = args.bioasq_data_path or ""
        bioasq_hash = hashlib.md5(bioasq_path.encode('utf-8')).hexdigest()[:8] if bioasq_path else "none"
        cache_key += f"_mix_bioasq_{args.bioasq_split}_r{args.bioasq_train_ratio:.2f}_{bioasq_hash}"
        if args.bioasq_max_train_samples > 0:
            cache_key += f"_cap{args.bioasq_max_train_samples}"
    processed_data_path = os.path.join(args.data_dir, cache_key)

    def map_with_fallback(dataset_obj, map_fn, batched=False, remove_columns=None, desc=None):
        map_kwargs = {
            'batched': batched,
            'desc': desc
        }
        if remove_columns is not None:
            map_kwargs['remove_columns'] = remove_columns
        if args.num_proc > 1:
            try:
                return dataset_obj.map(map_fn, num_proc=args.num_proc, **map_kwargs)
            except (PermissionError, OSError, RuntimeError) as exc:
                msg = str(exc).lower()
                mp_error = (
                    isinstance(exc, (PermissionError, OSError))
                    or "subprocesses has abruptly died" in msg
                    or "disable multiprocessing" in msg
                    or "thread creation failed" in msg
                    or "resource temporarily unavailable" in msg
                )
                if not mp_error:
                    raise
                print(f"Multiprocessing map failed ({exc}). Retrying in single-process mode.")
        return dataset_obj.map(map_fn, **map_kwargs)
    
    if os.path.exists(processed_data_path) and not args.reprocess_data:
        print(f"Loading preprocessed data from {processed_data_path}")
        tokenized_datasets = load_from_disk(processed_data_path)
        tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        def has_unmasked_padding(split_dataset, sample_size=128):
            if len(split_dataset) == 0:
                return False
            n = min(len(split_dataset), sample_size)
            for idx in range(n):
                sample = split_dataset[idx]
                if any(
                    int(attn) == 0 and int(label) != -100
                    for label, attn in zip(sample['labels'], sample['attention_mask'])
                ):
                    return True
            return False

        needs_padding_fix = any(has_unmasked_padding(tokenized_datasets[split]) for split in tokenized_datasets.keys())
        if not needs_padding_fix:
            return tokenized_datasets

        print("Detected legacy tokenized labels with unmasked padding. Fixing labels for this run...")

        def mask_padding_labels(batch):
            fixed_labels = []
            for labels, attention_mask in zip(batch['labels'], batch['attention_mask']):
                fixed_labels.append([
                    -100 if attn == 0 else int(label)
                    for label, attn in zip(labels, attention_mask)
                ])
            return {'labels': fixed_labels}

        tokenized_datasets = map_with_fallback(
            tokenized_datasets,
            mask_padding_labels,
            batched=True,
            desc="Masking padding tokens in labels"
        )
        tokenized_datasets = tokenized_datasets.filter(
            lambda x: any(int(t) != -100 for t in x['labels']),
            desc="Dropping samples with no supervised tokens"
        )
        tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        print("✓ Fixed tokenized dataset with padding labels masked")
        return tokenized_datasets
    
    # ------------------------------------------------------------------
    # BioASQ-only mode: skip PubMedQA entirely
    # ------------------------------------------------------------------
    if getattr(args, 'bioasq_only', False):
        if not args.bioasq_data_path:
            raise ValueError("--bioasq_only enabled but --bioasq_data_path is not provided")
        print(f"[BioASQ-only mode] Loading BioASQ dataset from: {args.bioasq_data_path}")
        bioasq_dataset = load_from_disk(args.bioasq_data_path)
        if isinstance(bioasq_dataset, DatasetDict):
            if args.bioasq_split in bioasq_dataset:
                bioasq_split = bioasq_dataset[args.bioasq_split]
            else:
                raise ValueError(f"BioASQ split '{args.bioasq_split}' not found")
        else:
            bioasq_split = bioasq_dataset

        print(f"Loaded {len(bioasq_split)} raw BioASQ samples")
        bioasq_formatted = map_with_fallback(
            bioasq_split,
            lambda x: format_bioasq_example(x, args.instruction_template),
            remove_columns=bioasq_split.column_names,
            desc="Formatting BioASQ examples"
        )
        bioasq_formatted = bioasq_formatted.filter(
            lambda x: x['response'] in ['yes', 'no', 'maybe'],
            desc="Filtering BioASQ to yes/no/maybe"
        )
        print(f"BioASQ valid yes/no/maybe samples: {len(bioasq_formatted)}")
        if len(bioasq_formatted) == 0:
            raise ValueError("No valid BioASQ yes/no/maybe samples after filtering.")

        # Split BioASQ into train/val (90/10)
        bioasq_split_data = bioasq_formatted.train_test_split(test_size=0.1, seed=args.seed)
        mixed_train = bioasq_split_data['train']
        pubmed_val = bioasq_split_data['test']
        print(f"BioASQ-only train: {len(mixed_train)}, val: {len(pubmed_val)}")

        dataset_dict = DatasetDict({
            'train': mixed_train,
            'validation': pubmed_val
        })
        # Skip the PubMedQA + mix_bioasq logic below
    else:
        # Normal path: load PubMedQA, optionally mix BioASQ
        pass

    if not getattr(args, 'bioasq_only', False):
        print("Loading PubMedQA dataset (pqa_labeled subset) from HuggingFace...")
        # Load only the pqa_labeled subset (1000 labeled examples)
        dataset = load_dataset('qiaojin/PubMedQA', 'pqa_labeled', split='train')
        print(f"Loaded {len(dataset)} samples from PubMedQA pqa_labeled")

        # Format examples into instruction-response format
        print("Formatting examples into instruction-response format...")
        formatted_dataset = map_with_fallback(
            dataset,
            lambda x: format_pubmedqa_example(x, args.instruction_template),
            remove_columns=dataset.column_names,
            desc="Formatting examples"
        )

        # Split into train (800) and validation (200)
        print("Splitting dataset into train and validation...")
        split_dataset = formatted_dataset.train_test_split(test_size=0.2, seed=args.seed)
        pubmed_train = split_dataset['train']
        pubmed_val = split_dataset['test']
        mixed_train = pubmed_train

        if args.mix_bioasq:
            if not args.bioasq_data_path:
                raise ValueError("--mix_bioasq enabled but --bioasq_data_path is not provided")

            print(f"Loading BioASQ dataset from local path: {args.bioasq_data_path}")
            bioasq_dataset = load_from_disk(args.bioasq_data_path)
            if isinstance(bioasq_dataset, DatasetDict):
                if args.bioasq_split in bioasq_dataset:
                    bioasq_split = bioasq_dataset[args.bioasq_split]
                else:
                    raise ValueError(f"BioASQ split '{args.bioasq_split}' not found in {args.bioasq_data_path}")
            else:
                bioasq_split = bioasq_dataset

            print(f"Loaded {len(bioasq_split)} raw BioASQ samples from split '{args.bioasq_split}'")
            bioasq_formatted = map_with_fallback(
                bioasq_split,
                lambda x: format_bioasq_example(x, args.instruction_template),
                remove_columns=bioasq_split.column_names,
                desc="Formatting BioASQ examples"
            )

            bioasq_formatted = bioasq_formatted.filter(
                lambda x: x['response'] in ['yes', 'no', 'maybe'],
                desc="Filtering BioASQ to yes/no/maybe"
            )
            print(f"BioASQ valid yes/no/maybe samples: {len(bioasq_formatted)}")

            if len(bioasq_formatted) == 0:
                raise ValueError("No valid BioASQ yes/no/maybe samples after filtering.")

            if args.bioasq_max_train_samples > 0:
                target_bioasq = args.bioasq_max_train_samples
            else:
                ratio = min(max(args.bioasq_train_ratio, 0.01), 0.99)
                target_bioasq = int(len(pubmed_train) * ratio / (1.0 - ratio))
                target_bioasq = max(1, target_bioasq)

            bioasq_shuffled = bioasq_formatted.shuffle(seed=args.seed)
            if len(bioasq_shuffled) >= target_bioasq:
                bioasq_sampled = bioasq_shuffled.select(range(target_bioasq))
            else:
                repeats = (target_bioasq + len(bioasq_shuffled) - 1) // len(bioasq_shuffled)
                repeated_parts = [bioasq_shuffled] * repeats
                bioasq_sampled = concatenate_datasets(repeated_parts).select(range(target_bioasq))

            mixed_train = concatenate_datasets([pubmed_train, bioasq_sampled]).shuffle(seed=args.seed)
            print(
                f"Mixed train set: PubMedQA={len(pubmed_train)}, "
                f"BioASQ={len(bioasq_sampled)}, Total={len(mixed_train)}"
            )

        dataset_dict = DatasetDict({
            'train': mixed_train,
            'validation': pubmed_val
        })

    print(f"Train samples: {len(dataset_dict['train'])}")
    print(f"Validation samples: {len(dataset_dict['validation'])}")

    # Tokenize the dataset
    def tokenize_function(examples):
        # Tokenize instruction + response
        full_encodings = tokenizer(
            examples['full_text'],
            padding='max_length',
            truncation=True,
            max_length=args.max_length
        )
        
        # Also tokenize instruction only to create labels (we only compute loss on response)
        instruction_encodings = tokenizer(
            examples['instruction'],
            padding='max_length',
            truncation=True,
            max_length=args.max_length
        )
        
        # Create labels: -100 for instruction tokens, actual tokens for response
        labels = []
        for i in range(len(full_encodings['input_ids'])):
            label = full_encodings['input_ids'][i].copy()
            instruction_length = len([t for t in instruction_encodings['input_ids'][i] if t != tokenizer.pad_token_id])
            
            # Set instruction tokens to -100 (ignore in loss)
            for j in range(instruction_length):
                label[j] = -100

            # Set padding tokens to -100 so loss is not dominated by pad prediction
            for j, attn in enumerate(full_encodings['attention_mask'][i]):
                if attn == 0:
                    label[j] = -100
            
            labels.append(label)
        
        full_encodings['labels'] = labels
        return full_encodings
    
    # Save a reference for printing examples later
    _raw_dataset_dict = dataset_dict

    print("Tokenizing dataset...")
    tokenized_datasets = map_with_fallback(
        dataset_dict,
        tokenize_function,
        batched=True,
        remove_columns=dataset_dict['train'].column_names,
        desc="Tokenizing"
    )

    tokenized_datasets = tokenized_datasets.filter(
        lambda x: any(int(t) != -100 for t in x['labels']),
        desc="Dropping samples with no supervised tokens"
    )
    
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    # Save processed data
    os.makedirs(args.data_dir, exist_ok=True)
    print(f"Saving processed data to {processed_data_path}")
    tokenized_datasets.save_to_disk(processed_data_path)
    
    # Print example
    print("\n" + "="*50)
    print("Example formatted sample:")
    print("="*50)
    try:
        example = _raw_dataset_dict['train'][0]
        print(f"Instruction:\n{example.get('instruction', 'N/A')}\n")
        print(f"Response: {example.get('response', 'N/A')}")
    except Exception:
        print("(Could not display example)")
    print("="*50 + "\n")
    
    return tokenized_datasets


def evaluate_model(model, dataloader, device, max_batches=None):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
            
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(inputs)
            logits = outputs.logits
            
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # Compute loss (ignore -100 labels)
            valid_tokens = (shift_labels != -100).sum()
            if valid_tokens.item() == 0:
                continue
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            if not torch.isfinite(loss):
                continue
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')
    
    return avg_loss, perplexity


def train_mamba2_sft(args):
    """Main SFT training function"""
    
    # Set device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.set_device(args.gpu_id)
        print(f"Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Load tokenizer
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")
    
    # Prefer dedicated padding token if available; fallback to EOS.
    if tokenizer.pad_token is None:
        vocab = tokenizer.get_vocab()
        if "<|padding|>" in vocab:
            tokenizer.pad_token = "<|padding|>"
        else:
            tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare data
    print("\n" + "="*50)
    print("Preparing SFT dataset...")
    print("="*50)
    tokenized_datasets = prepare_pubmedqa_data(args, tokenizer)
    
    train_dataset = tokenized_datasets['train']
    val_dataset = tokenized_datasets['validation']
    
    # Create data loaders
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size)
    
    # Load model
    print(f"\nLoading Mamba2 model from: {args.model_path}")
    model = MambaLMHeadModel.from_pretrained(args.model_path)
    model.to(device)
    print(f"Model loaded successfully")
    
    # Inject LoRA if requested
    lora_params = []
    if args.use_lora:
        print(f"\nInjecting LoRA adapters (rank={args.lora_rank}, alpha={args.lora_alpha})...")
        print(f"Target modules: {args.lora_target_modules}")
        lora_params = inject_lora_to_mamba(
            model, 
            rank=args.lora_rank, 
            alpha=args.lora_alpha,
            target_modules=args.lora_target_modules
        )
        print(f"✓ Injected LoRA to {len(lora_params)//2} modules")
        
        # Only optimize LoRA parameters
        trainable_params = lora_params
    else:
        # Full fine-tuning
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_count = sum(p.numel() for p in trainable_params)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params_count:,}")
    print(f"Trainable %: {100 * trainable_params_count / total_params:.2f}%")
    
    # Optimizer
    optimizer = AdamW(
        trainable_params,
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay
    )
    
    # Calculate training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.accumulation_steps)
    num_training_steps = num_update_steps_per_epoch * args.num_epochs
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    
    # Scheduler
    if args.scheduler == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    
    print(f"\nTraining Configuration:")
    print(f"  Number of epochs: {args.num_epochs}")
    print(f"  Optimizer steps per epoch: {num_update_steps_per_epoch}")
    print(f"  Total training steps: {num_training_steps}")
    print(f"  Warmup steps: {num_warmup_steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation steps: {args.accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.accumulation_steps}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Scheduler: {args.scheduler}")
    
    # Mixed precision training
    if args.fp16 and args.bf16:
        raise ValueError("Choose only one precision mode: --fp16 or --bf16")

    amp_dtype = None
    use_amp = False
    if device.type == 'cuda':
        if args.bf16:
            use_amp = True
            amp_dtype = torch.bfloat16
        elif args.fp16:
            use_amp = True
            amp_dtype = torch.float16
    else:
        if args.fp16 or args.bf16:
            print("Warning: mixed precision enabled but CUDA is unavailable. Running in full precision.")

    # GradScaler is only needed for fp16, not bf16.
    if use_amp and amp_dtype == torch.float16:
        scaler = TorchGradScaler("cuda") if USE_TORCH_AMP else TorchGradScaler()
    else:
        scaler = None
    
    # Prepare output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lora_suffix = f"_lora_r{args.lora_rank}" if args.use_lora else "_full"
    output_dir = os.path.join(args.output_dir, f"biomamba2_sft_{args.model_name}{lora_suffix}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Tensorboard
    log_dir = os.path.join(args.log_dir, f"biomamba2_sft_{args.model_name}{lora_suffix}_{timestamp}")
    writer = SummaryWriter(log_dir=log_dir)
    
    # Initialize Wandb
    use_wandb = args.use_wandb and WANDB_AVAILABLE
    if use_wandb:
        run_name = args.wandb_run_name or f"biomamba2_sft_{args.model_name}{lora_suffix}_{timestamp}"
        dataset_name = 'pubmedqa_pqa_labeled+bioasq' if args.mix_bioasq else 'pubmedqa_pqa_labeled'
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            tags=[args.model_name, 'pubmedqa', 'bioasq' if args.mix_bioasq else 'no-bioasq', 'sft', 'lora' if args.use_lora else 'full'],
            config={
                'model_name': args.model_name,
                'model_path': args.model_path,
                'use_lora': args.use_lora,
                'lora_rank': args.lora_rank if args.use_lora else None,
                'lora_alpha': args.lora_alpha if args.use_lora else None,
                'total_params': total_params,
                'trainable_params': trainable_params_count,
                'dataset': dataset_name,
                'mix_bioasq': args.mix_bioasq,
                'bioasq_data_path': args.bioasq_data_path,
                'bioasq_split': args.bioasq_split if args.mix_bioasq else None,
                'bioasq_train_ratio': args.bioasq_train_ratio if args.mix_bioasq else None,
                'train_samples': len(train_dataset),
                'val_samples': len(val_dataset),
                'batch_size': args.batch_size,
                'accumulation_steps': args.accumulation_steps,
                'num_epochs': args.num_epochs,
                'learning_rate': args.lr,
                'weight_decay': args.weight_decay,
                'max_length': args.max_length,
            }
        )
        print(f"✓ Wandb initialized: {wandb.run.url}")
    
    # Save training arguments
    with open(os.path.join(output_dir, 'training_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Log directory: {log_dir}")
    print("\n" + "="*50)
    print("Starting SFT training...")
    print("="*50 + "\n")
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    best_step = -1
    model.train()
    optimizer.zero_grad()

    def save_model_artifacts(save_path):
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        if args.use_lora:
            lora_config = {
                'rank': args.lora_rank,
                'alpha': args.lora_alpha,
                'target_modules': args.lora_target_modules
            }
            with open(os.path.join(save_path, 'lora_config.json'), 'w') as f:
                json.dump(lora_config, f, indent=2)
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        epoch_loss = 0.0
        
        progress_bar = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc=f"Epoch {epoch + 1}"
        )
        
        for batch_idx, batch in progress_bar:
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass with mixed precision
            if use_amp:
                autocast_kwargs = {'device_type': 'cuda'} if USE_TORCH_AMP else {}
                if amp_dtype is not None:
                    autocast_kwargs['dtype'] = amp_dtype
                with torch_autocast(**autocast_kwargs):
                    outputs = model(inputs)
                    logits = outputs.logits
                    
                    # Shift for next-token prediction
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()

                    valid_tokens = (shift_labels != -100).sum()
                    if valid_tokens.item() == 0:
                        continue

                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100
                    )
                    if not torch.isfinite(loss):
                        print("Warning: non-finite loss encountered; skipping batch")
                        continue
                    loss = loss / args.accumulation_steps

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                outputs = model(inputs)
                logits = outputs.logits
                
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                valid_tokens = (shift_labels != -100).sum()
                if valid_tokens.item() == 0:
                    continue

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
                if not torch.isfinite(loss):
                    print("Warning: non-finite loss encountered; skipping batch")
                    continue
                loss = loss / args.accumulation_steps
                
                loss.backward()
            
            epoch_loss += loss.item() * args.accumulation_steps
            
            # Gradient accumulation
            should_step = (
                (batch_idx + 1) % args.accumulation_steps == 0
                or (batch_idx + 1) == len(train_dataloader)
            )

            if should_step:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Logging
                if global_step % args.logging_steps == 0 or global_step == 1:
                    current_lr = scheduler.get_last_lr()[0]
                    train_loss = loss.item() * args.accumulation_steps
                    
                    writer.add_scalar('Train/Loss', train_loss, global_step)
                    writer.add_scalar('Train/LearningRate', current_lr, global_step)
                    
                    if use_wandb:
                        wandb.log({
                            'train/loss': train_loss,
                            'train/learning_rate': current_lr,
                            'train/epoch': epoch + 1,
                            'train/step': global_step,
                        }, step=global_step)
                    
                    progress_bar.set_postfix({
                        'loss': f'{train_loss:.4f}',
                        'lr': f'{current_lr:.2e}'
                    })
                
                # Evaluation
                if global_step % args.eval_steps == 0:
                    print(f"\nEvaluating at step {global_step}...")
                    avg_loss, perplexity = evaluate_model(model, val_dataloader, device)
                    
                    writer.add_scalar('Eval/Loss', avg_loss, global_step)
                    writer.add_scalar('Eval/Perplexity', perplexity, global_step)
                    
                    if use_wandb:
                        wandb.log({
                            'eval/loss': avg_loss,
                            'eval/perplexity': perplexity,
                        }, step=global_step)
                    
                    print(f"Validation - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
                    
                    # Save best model
                    if avg_loss < best_val_loss:
                        best_val_loss = avg_loss
                        best_step = global_step
                        best_model_path = os.path.join(output_dir, 'best_model')
                        print(f"New best model! Saving to {best_model_path}")
                        save_model_artifacts(best_model_path)
                        
                        if use_wandb:
                            wandb.run.summary['best_val_loss'] = best_val_loss
                            wandb.run.summary['best_step'] = global_step
                    
                    model.train()
                
                # Save checkpoint
                if global_step % args.save_steps == 0:
                    checkpoint_path = os.path.join(output_dir, f'checkpoint-{global_step}')
                    print(f"\nSaving checkpoint to {checkpoint_path}")
                    save_model_artifacts(checkpoint_path)
        
        # End of epoch evaluation
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"\nEpoch {epoch + 1} completed. Average training loss: {avg_epoch_loss:.4f}")
        
        print(f"End of epoch validation...")
        avg_loss, perplexity = evaluate_model(model, val_dataloader, device)
        print(f"Validation - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
        
        writer.add_scalar('Eval/EpochLoss', avg_loss, epoch)
        writer.add_scalar('Eval/EpochPerplexity', perplexity, epoch)
        
        if use_wandb:
            wandb.log({
                'epoch/train_loss': avg_epoch_loss,
                'epoch/val_loss': avg_loss,
                'epoch/val_perplexity': perplexity,
                'epoch/number': epoch + 1,
            }, step=global_step)

        # Save best model based on end-of-epoch validation as a fallback
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            best_step = global_step
            best_model_path = os.path.join(output_dir, 'best_model')
            print(f"New best model at epoch end! Saving to {best_model_path}")
            save_model_artifacts(best_model_path)

            if use_wandb:
                wandb.run.summary['best_val_loss'] = best_val_loss
                wandb.run.summary['best_step'] = best_step
    
    # Final save
    final_model_path = os.path.join(output_dir, 'final_model')
    print(f"\nTraining completed! Saving final model to {final_model_path}")
    save_model_artifacts(final_model_path)
    
    # Final evaluation
    print("\nFinal evaluation on validation set...")
    avg_loss, perplexity = evaluate_model(model, val_dataloader, device)
    print(f"Final Validation Loss: {avg_loss:.4f}")
    print(f"Final Validation Perplexity: {perplexity:.4f}")

    if avg_loss < best_val_loss:
        best_val_loss = avg_loss
        best_step = global_step
        best_model_path = os.path.join(output_dir, 'best_model')
        print(f"Final model is new best! Saving to {best_model_path}")
        save_model_artifacts(best_model_path)
    
    # Save final metrics
    final_metrics = {
        'final_val_loss': avg_loss,
        'final_val_perplexity': perplexity,
        'best_val_loss': best_val_loss,
        'best_step': best_step,
        'total_steps': global_step
    }
    
    with open(os.path.join(output_dir, 'final_metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    if use_wandb:
        wandb.run.summary.update(final_metrics)
        wandb.finish()
        print("✓ Wandb run completed")
    
    writer.close()
    print(f"\nAll outputs saved to: {output_dir}")
    
    return model, tokenizer


def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Train model
    model, tokenizer = train_mamba2_sft(args)
    
    print("\n" + "="*50)
    print("SFT training completed successfully!")
    print("="*50)


if __name__ == "__main__":
    main()
