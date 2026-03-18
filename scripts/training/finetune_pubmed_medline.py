"""
Mamba/Mamba2 Domain Post-Training on PubMed-MEDLINE Dataset
Dataset: https://huggingface.co/datasets/cyrilzakka/pubmed-medline

Improvements:
- Support for both Mamba1 and Mamba2 models
- Layer-wise learning rate decay
- Exponential Moving Average (EMA)
- Label smoothing
- Cosine with restarts scheduler
- Gradient checkpointing for memory efficiency
- Data augmentation (optional MLM-style masking)
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
import copy
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR
from datasets import load_dataset, load_from_disk, DatasetDict
from mamba_ssm import MambaLMHeadModel
import argparse
from datetime import datetime
from collections import OrderedDict

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")


# ============================================================================
# Model Configuration: Mamba1 vs Mamba2
# ============================================================================
MAMBA1_MODELS = {
    'mamba-130m': {'batch_size': 32, 'path': 'state-spaces/mamba-130m'},
    'mamba-370m': {'batch_size': 16, 'path': 'state-spaces/mamba-370m'},
    'mamba-790m': {'batch_size': 8, 'path': 'state-spaces/mamba-790m'},
    'mamba-1.4b': {'batch_size': 4, 'path': 'state-spaces/mamba-1.4b'},
    'mamba-2.8b': {'batch_size': 2, 'path': 'state-spaces/mamba-2.8b'},
}

MAMBA2_MODELS = {
    'mamba2-130m': {'batch_size': 32, 'path': 'state-spaces/mamba2-130m'},
    'mamba2-370m': {'batch_size': 16, 'path': 'state-spaces/mamba2-370m'},
    'mamba2-780m': {'batch_size': 8, 'path': 'state-spaces/mamba2-780m'},
    'mamba2-1.3b': {'batch_size': 4, 'path': 'state-spaces/mamba2-1.3b'},
    'mamba2-2.7b': {'batch_size': 2, 'path': 'state-spaces/mamba2-2.7b'},
}

ALL_MODELS = {**MAMBA1_MODELS, **MAMBA2_MODELS}


def is_mamba1_model(model_name):
    """Check if the model is Mamba1 (not Mamba2)"""
    return model_name in MAMBA1_MODELS or model_name.startswith('mamba-')


def is_mamba2_model(model_name):
    """Check if the model is Mamba2"""
    return model_name in MAMBA2_MODELS or model_name.startswith('mamba2-')


# ============================================================================
# Exponential Moving Average (EMA)
# ============================================================================
class EMA:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register()

    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply EMA weights for evaluation"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original weights after evaluation"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# ============================================================================
# Label Smoothing Cross Entropy Loss
# ============================================================================
class LabelSmoothingCrossEntropy(torch.nn.Module):
    """Cross entropy loss with label smoothing"""
    def __init__(self, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits, targets):
        n_classes = logits.size(-1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Create smoothed targets
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        loss = (-true_dist * log_probs).sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ============================================================================
# WSD (Warmup-Stable-Decay) Learning Rate Scheduler
# ============================================================================
def get_wsd_schedule(optimizer, num_warmup_steps, num_stable_steps, num_decay_steps, 
                     min_lr_ratio=0.1, last_epoch=-1):
    """
    Warmup → Stable → Decay scheduler.
    
    LR schedule:
      - [0, warmup]:         linear warmup from 0 → peak_lr
      - [warmup, stable]:    constant at peak_lr
      - [stable, end]:       cosine decay from peak_lr → min_lr_ratio * peak_lr
    
    This is the recommended schedule for continued pre-training, where:
      - Warmup lets the optimizer settle
      - Stable plateau is the main training phase
      - Gentle decay at the end helps convergence
    """
    total_steps = num_warmup_steps + num_stable_steps + num_decay_steps
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step < num_warmup_steps + num_stable_steps:
            # Stable phase - constant LR
            return 1.0
        else:
            # Cosine decay phase
            decay_step = current_step - num_warmup_steps - num_stable_steps
            decay_progress = float(decay_step) / float(max(1, num_decay_steps))
            # Cosine decay from 1.0 to min_lr_ratio
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


# ============================================================================
# Layer-wise Learning Rate Decay
# ============================================================================
def get_layer_wise_lr_decay_params(model, lr, weight_decay, layer_decay=0.75):
    """
    Apply layer-wise learning rate decay.
    Deeper layers get smaller learning rates.
    """
    param_groups = []
    
    # Get all named parameters
    named_params = list(model.named_parameters())
    
    # Try to identify layer depth from parameter names
    # For Mamba models, we look for patterns like 'layers.0', 'layers.1', etc.
    layer_params = {}
    other_params = []
    
    num_layers = 0
    for name, param in named_params:
        if not param.requires_grad:
            continue
        
        # Try to extract layer number
        layer_num = None
        if 'layers.' in name or 'backbone.layers.' in name:
            try:
                # Extract layer number from name like "backbone.layers.5.mixer..."
                parts = name.split('.')
                for i, part in enumerate(parts):
                    if part == 'layers' and i + 1 < len(parts):
                        layer_num = int(parts[i + 1])
                        num_layers = max(num_layers, layer_num + 1)
                        break
            except (ValueError, IndexError):
                pass
        
        if layer_num is not None:
            if layer_num not in layer_params:
                layer_params[layer_num] = []
            layer_params[layer_num].append((name, param))
        else:
            other_params.append((name, param))
    
    # Apply layer-wise decay
    if num_layers > 0:
        for layer_num in sorted(layer_params.keys()):
            # Layer 0 is closest to input, gets smallest LR
            # Last layer is closest to output, gets largest LR
            layer_lr = lr * (layer_decay ** (num_layers - layer_num - 1))
            
            param_groups.append({
                'params': [p for _, p in layer_params[layer_num]],
                'lr': layer_lr,
                'weight_decay': weight_decay,
                'layer': layer_num
            })
        
        print(f"Applied layer-wise LR decay across {num_layers} layers")
        print(f"  Layer 0 LR: {lr * (layer_decay ** (num_layers - 1)):.2e}")
        print(f"  Layer {num_layers-1} LR: {lr:.2e}")
    
    # Add other parameters (embeddings, LM head, etc.) with base LR
    if other_params:
        param_groups.append({
            'params': [p for _, p in other_params],
            'lr': lr,
            'weight_decay': weight_decay,
            'layer': 'other'
        })
    
    # If no layers found, fall back to simple param grouping
    if not param_groups:
        param_groups = [{'params': [p for _, p in named_params if p.requires_grad], 
                        'lr': lr, 'weight_decay': weight_decay}]
    
    return param_groups


def parse_args():
    parser = argparse.ArgumentParser(description='Mamba/Mamba2 Domain Post-Training on PubMed-MEDLINE')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='mamba2-130m', 
                        choices=list(ALL_MODELS.keys()),
                        help='Mamba model to use (supports both Mamba1 and Mamba2)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to pretrained model (if continuing from checkpoint)')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory to save processed data')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--use_title', action='store_true', default=True,
                        help='Include title in training text')
    parser.add_argument('--test_size', type=float, default=0.1,
                        help='Test set ratio')
    parser.add_argument('--num_proc', type=int, default=16,
                        help='Number of processes for data preprocessing')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (auto-determined if None)')
    parser.add_argument('--accumulation_steps', type=int, default=8,
                        help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=5e-6,
                        help='Learning rate (default: 5e-6, suitable for continued pretraining)')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--num_training_steps', type=int, default=None,
                        help='Total training steps (overrides num_epochs if set)')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup ratio')
    parser.add_argument('--scheduler', type=str, default='wsd',
                        choices=['linear', 'cosine', 'cosine_restarts', 'wsd'],
                        help='LR scheduler: linear, cosine, cosine_restarts, wsd (warmup-stable-decay)')
    parser.add_argument('--num_restarts', type=int, default=3,
                        help='Number of restarts for cosine_restarts scheduler')
    parser.add_argument('--stable_ratio', type=float, default=0.7,
                        help='Ratio of total steps for stable phase in wsd scheduler (default: 0.7)')
    parser.add_argument('--decay_ratio', type=float, default=0.2,
                        help='Ratio of total steps for decay phase in wsd scheduler (default: 0.2)')
    parser.add_argument('--min_lr_ratio', type=float, default=0.1,
                        help='Minimum LR as a ratio of peak LR at end of decay (default: 0.1)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping')
    
    # Advanced training techniques
    parser.add_argument('--use_ema', action='store_true',
                        help='Use Exponential Moving Average')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='EMA decay rate')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing (0.0 to disable, 0.1 recommended)')
    parser.add_argument('--layer_wise_lr_decay', type=float, default=1.0,
                        help='Layer-wise LR decay (1.0 to disable, 0.75-0.9 recommended)')
    parser.add_argument('--data_augmentation', action='store_true',
                        help='Enable MLM-style data augmentation')
    parser.add_argument('--mlm_probability', type=float, default=0.15,
                        help='Masking probability for data augmentation')
    
    # Logging and saving
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--log_dir', type=str, default='./runs',
                        help='Tensorboard log directory')
    parser.add_argument('--logging_steps', type=int, default=100,
                        help='Log every N steps')
    parser.add_argument('--eval_steps', type=int, default=1000,
                        help='Evaluate every N steps')
    parser.add_argument('--save_steps', type=int, default=2000,
                        help='Save checkpoint every N steps')
    parser.add_argument('--eval_samples', type=int, default=100,
                        help='Number of batches for evaluation')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='Use mixed precision training (fp16)')
    parser.add_argument('--bf16', action='store_true', default=False,
                        help='Use bf16 mixed precision (recommended for H100/A100, overrides fp16)')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='Use torch.compile for faster training (PyTorch 2.0+)')
    
    # Data loading
    parser.add_argument('--reprocess_data', action='store_true',
                        help='Force reprocessing of data')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Wandb arguments
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='biomamba2-training',
                        help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Wandb entity (username or team)')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Wandb run name (default: auto-generated)')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=None,
                        help='Wandb tags for the run')
    parser.add_argument('--wandb_notes', type=str, default=None,
                        help='Wandb notes for the run')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_pubmed_medline_data(args, tokenizer):
    """
    Load and preprocess the PubMed-MEDLINE dataset
    """
    processed_data_path = os.path.join(args.data_dir, f'pubmed_medline_tokenized_{args.max_length}')
    
    if os.path.exists(processed_data_path) and not args.reprocess_data:
        print(f"Loading preprocessed data from {processed_data_path}")
        return load_from_disk(processed_data_path)
    
    print("Loading PubMed-MEDLINE dataset from HuggingFace...")
    # Load the dataset
    dataset = load_dataset('cyrilzakka/pubmed-medline', split='train')
    print(f"Loaded {len(dataset)} samples from PubMed-MEDLINE")
    
    # Filter out samples with empty content
    print("Filtering samples with empty content...")
    dataset = dataset.filter(
        lambda x: x['content'] is not None and len(x['content'].strip()) > 0,
        num_proc=args.num_proc
    )
    print(f"After filtering: {len(dataset)} samples")
    
    # Create training text (title + content)
    def create_text(example):
        if args.use_title and example['title']:
            # Combine title and content with a separator
            text = f"{example['title']}\n\n{example['content']}"
        else:
            text = example['content']
        return {'text': text}
    
    print("Creating training text from title and content...")
    dataset = dataset.map(create_text, num_proc=args.num_proc)
    
    # Remove unnecessary columns to save memory
    columns_to_remove = [col for col in dataset.column_names if col != 'text']
    dataset = dataset.remove_columns(columns_to_remove)
    
    # Deduplication to prevent data leakage
    print(f"Before deduplication: {len(dataset)} samples")
    try:
        import pandas as pd
        from datasets import Dataset
        df = dataset.to_pandas()
        initial_count = len(df)
        df.drop_duplicates(subset=['text'], inplace=True)
        dedup_count = len(df)
        
        if initial_count != dedup_count:
            print(f"⚠️ Removed {initial_count - dedup_count} duplicate samples to prevent leakage!")
            dataset = Dataset.from_pandas(df)
            # Remove pandas index column if created
            if '__index_level_0__' in dataset.column_names:
                dataset = dataset.remove_columns(['__index_level_0__'])
        else:
            print("No duplicates found.")
            
    except ImportError:
        print("Warning: pandas not installed, skipping deduplication. Install pandas for stricter data splitting.")
    except Exception as e:
        print(f"Warning: Deduplication failed: {e}")
    
    print(f"After deduplication: {len(dataset)} samples")
    
    # Split into train and test
    print(f"Splitting dataset (test_size={args.test_size})...")
    split_dataset = dataset.train_test_split(test_size=args.test_size, seed=args.seed)
    
    dataset_dict = DatasetDict({
        'train': split_dataset['train'],
        'test': split_dataset['test']
    })
    
    print(f"Train samples: {len(dataset_dict['train'])}")
    print(f"Test samples: {len(dataset_dict['test'])}")
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples['text'], 
            padding='max_length', 
            truncation=True, 
            max_length=args.max_length
        )
    
    print("Tokenizing dataset...")
    tokenized_datasets = dataset_dict.map(
        tokenize_function, 
        batched=True, 
        num_proc=args.num_proc,
        remove_columns=['text']
    )
    
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    # Save processed data
    os.makedirs(args.data_dir, exist_ok=True)
    print(f"Saving processed data to {processed_data_path}")
    tokenized_datasets.save_to_disk(processed_data_path)
    
    return tokenized_datasets


def evaluate_model(model, dataloader, device, max_batches=None, amp_dtype=None):
    """Evaluate model on test set (with optional mixed precision for speed)"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
            
            inputs = batch['input_ids'].to(device, non_blocking=True)
            labels = batch['input_ids'].to(device, non_blocking=True)
            
            if amp_dtype is not None:
                with autocast(device_type='cuda', dtype=amp_dtype):
                    outputs = model(inputs)
                    logits = outputs.logits
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    loss_fct = torch.nn.CrossEntropyLoss()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            else:
                outputs = model(inputs)
                logits = outputs.logits
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')
    
    return avg_loss, perplexity


def train_mamba2(args):
    """Main training function for both Mamba1 and Mamba2 models"""
    
    # Detect model type
    model_type = "Mamba1" if is_mamba1_model(args.model_name) else "Mamba2"
    print(f"\n{'='*50}")
    print(f"Detected model type: {model_type}")
    print(f"{'='*50}")
    
    # Set device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.set_device(args.gpu_id)
        print(f"Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Load tokenizer
    print(f"Loading tokenizer for {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare data
    print("\n" + "="*50)
    print("Preparing PubMed-MEDLINE dataset...")
    print("="*50)
    tokenized_datasets = prepare_pubmed_medline_data(args, tokenizer)
    
    train_dataset = tokenized_datasets['train']
    test_dataset = tokenized_datasets['test']
    
    # Determine batch size if not specified
    if args.batch_size is None:
        if args.model_name in ALL_MODELS:
            batch_size = ALL_MODELS[args.model_name]['batch_size']
        else:
            batch_size = 16
        print(f"Auto-determined batch_size: {batch_size}")
    else:
        batch_size = args.batch_size
    
    effective_batch_size = batch_size * args.accumulation_steps
    print(f"Effective batch size: {effective_batch_size}")
    
    # Create data loaders
    # num_workers: overlap CPU data loading with GPU compute
    # pin_memory: faster host→device transfer on CUDA
    # persistent_workers: keep workers alive between epochs (avoid re-spawn overhead)
    num_workers = min(args.num_proc, 8)  # Cap at 8, diminishing returns beyond that
    use_pin_memory = (args.device == 'cuda')
    
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size,
        num_workers=num_workers, pin_memory=use_pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    test_dataloader = DataLoader(
        test_dataset, shuffle=False, batch_size=batch_size,
        num_workers=num_workers, pin_memory=use_pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    print(f"DataLoader: num_workers={num_workers}, pin_memory={use_pin_memory}")
    
    # Load model - supports both Mamba1 and Mamba2
    print(f"\nLoading {model_type} model: {args.model_name}")
    if args.model_path:
        print(f"Loading from checkpoint: {args.model_path}")
        model = MambaLMHeadModel.from_pretrained(args.model_path)
    else:
        # Get model path from configuration
        if args.model_name in ALL_MODELS:
            model_hf_path = ALL_MODELS[args.model_name]['path']
        else:
            model_hf_path = f"state-spaces/{args.model_name}"
        print(f"Loading from HuggingFace: {model_hf_path}")
        model = MambaLMHeadModel.from_pretrained(model_hf_path)
    
    model.to(device)
    print(f"Model loaded successfully")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ---- Mixed Precision Configuration ----
    # bf16 overrides fp16 (bf16 is better on H100/A100: no scaling needed, larger dynamic range)
    if args.bf16:
        if not torch.cuda.is_bf16_supported():
            print("WARNING: bf16 requested but not supported on this GPU. Falling back to fp16.")
            args.bf16 = False
        else:
            args.fp16 = False  # bf16 takes priority
            print("Using bf16 mixed precision (native on Ampere+/Hopper GPUs)")
    
    # ---- torch.compile for kernel fusion ----
    if args.compile:
        try:
            print("Compiling model with torch.compile (this may take 1-2 minutes the first time)...")
            model = torch.compile(model)
            print("Model compiled successfully - expect ~10-30% speedup during training")
        except Exception as e:
            print(f"WARNING: torch.compile failed ({e}), continuing without compilation")
    
    # Initialize EMA if enabled
    ema = None
    if args.use_ema:
        ema = EMA(model, decay=args.ema_decay)
        print(f"EMA enabled with decay={args.ema_decay}")
    
    # Setup loss function with optional label smoothing
    if args.label_smoothing > 0:
        loss_fct = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
        print(f"Label smoothing enabled with smoothing={args.label_smoothing}")
    else:
        loss_fct = torch.nn.CrossEntropyLoss()
    
    # Optimizer with optional layer-wise LR decay
    if args.layer_wise_lr_decay < 1.0:
        param_groups = get_layer_wise_lr_decay_params(
            model, args.lr, args.weight_decay, args.layer_wise_lr_decay
        )
        optimizer = AdamW(param_groups, betas=(0.9, 0.95))
    else:
        optimizer = AdamW(
            model.parameters(), 
            lr=args.lr, 
            betas=(0.9, 0.95), 
            weight_decay=args.weight_decay
        )
    
    # Calculate training steps
    # IMPORTANT: Account for gradient accumulation!
    # The scheduler steps once per optimizer step, not per batch
    num_batches_per_epoch = len(train_dataloader)
    num_optimizer_steps_per_epoch = math.ceil(num_batches_per_epoch / args.accumulation_steps)
    
    if args.num_training_steps:
        # User specified total optimizer steps
        num_training_steps = args.num_training_steps
        num_epochs = math.ceil(num_training_steps / num_optimizer_steps_per_epoch)
    else:
        # Calculate from epochs
        num_epochs = args.num_epochs
        num_training_steps = num_optimizer_steps_per_epoch * num_epochs
    
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    
    print(f"\n[LR Scheduler Configuration]")
    print(f"  Batches per epoch: {num_batches_per_epoch}")
    print(f"  Gradient accumulation steps: {args.accumulation_steps}")
    print(f"  Optimizer steps per epoch: {num_optimizer_steps_per_epoch}")
    print(f"  Total optimizer steps: {num_training_steps}")
    print(f"  Warmup steps: {num_warmup_steps}")
    
    # Scheduler creation
    if args.scheduler == 'wsd':
        # Warmup-Stable-Decay: best for continued pre-training
        # warmup_ratio is already used for num_warmup_steps
        remaining_steps = num_training_steps - num_warmup_steps
        # stable_ratio and decay_ratio are relative to the remaining (non-warmup) steps
        num_stable_steps = int(args.stable_ratio / (args.stable_ratio + args.decay_ratio) * remaining_steps)
        num_decay_steps = remaining_steps - num_stable_steps
        
        scheduler = get_wsd_schedule(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_stable_steps=num_stable_steps,
            num_decay_steps=num_decay_steps,
            min_lr_ratio=args.min_lr_ratio,
        )
        print(f"  Scheduler: WSD (Warmup-Stable-Decay)")
        print(f"    Phase 1 - Warmup:  {num_warmup_steps} steps (LR: 0 → {args.lr:.2e})")
        print(f"    Phase 2 - Stable:  {num_stable_steps} steps (LR: {args.lr:.2e} constant)")
        print(f"    Phase 3 - Decay:   {num_decay_steps} steps (LR: {args.lr:.2e} → {args.lr * args.min_lr_ratio:.2e})")
        print(f"    Min LR ratio: {args.min_lr_ratio}")
        
    elif args.scheduler == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=num_warmup_steps, 
            num_training_steps=num_training_steps
        )
        print(f"  Scheduler: Cosine with warmup")
        print(f"  After {num_warmup_steps} warmup steps, LR will cosine decay to 0")
        
    elif args.scheduler == 'cosine_restarts':
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=args.num_restarts
        )
        print(f"  Scheduler: Cosine with {args.num_restarts} hard restarts")
        
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=num_warmup_steps, 
            num_training_steps=num_training_steps
        )
        print(f"  Scheduler: Linear with warmup")
    
    print(f"\nTraining Configuration:")
    print(f"\n[Training Configuration]")
    print(f"  Model type: {model_type}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  EMA: {'enabled' if args.use_ema else 'disabled'}")
    print(f"  Label smoothing: {args.label_smoothing}")
    print(f"  Layer-wise LR decay: {args.layer_wise_lr_decay}")
    print(f"  Data augmentation: {'enabled' if args.data_augmentation else 'disabled'}")
    
    # Mixed precision training
    # bf16 does NOT need GradScaler (wider dynamic range means no loss scaling needed)
    use_amp = args.fp16 or args.bf16
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    scaler = GradScaler() if args.fp16 else None  # Only fp16 needs scaler
    
    print(f"\n[Mixed Precision]")
    if args.bf16:
        print(f"  Mode: bf16 (no scaler needed)")
    elif args.fp16:
        print(f"  Mode: fp16 (with GradScaler)")
    else:
        print(f"  Mode: fp32 (full precision)")
    
    # Prepare output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_prefix = "biomamba" if is_mamba1_model(args.model_name) else "biomamba2"
    output_dir = os.path.join(args.output_dir, f"{model_prefix}_{args.model_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Tensorboard
    log_dir = os.path.join(args.log_dir, f"{model_prefix}_{args.model_name}_{timestamp}")
    writer = SummaryWriter(log_dir=log_dir)
    
    # Initialize Wandb
    use_wandb = args.use_wandb and WANDB_AVAILABLE
    if use_wandb:
        # Set run name
        run_name = args.wandb_run_name or f"{model_prefix}_{args.model_name}_{timestamp}"
        
        # Initialize wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            tags=args.wandb_tags or [args.model_name, model_type, 'pubmed-medline', 'domain-training'],
            notes=args.wandb_notes,
            config={
                # Model config
                'model_name': args.model_name,
                'model_type': model_type,
                'total_params': total_params,
                'trainable_params': trainable_params,
                
                # Data config
                'dataset': 'pubmed-medline',
                'max_length': args.max_length,
                'use_title': args.use_title,
                'train_samples': len(train_dataset),
                'test_samples': len(test_dataset),
                
                # Training config
                'batch_size': batch_size,
                'accumulation_steps': args.accumulation_steps,
                'effective_batch_size': effective_batch_size,
                'num_epochs': num_epochs,
                'num_training_steps': num_training_steps,
                'learning_rate': args.lr,
                'weight_decay': args.weight_decay,
                'warmup_ratio': args.warmup_ratio,
                'warmup_steps': num_warmup_steps,
                'scheduler': args.scheduler,
                'max_grad_norm': args.max_grad_norm,
                'fp16': args.fp16,
                
                # Advanced techniques
                'use_ema': args.use_ema,
                'ema_decay': args.ema_decay if args.use_ema else None,
                'label_smoothing': args.label_smoothing,
                'layer_wise_lr_decay': args.layer_wise_lr_decay,
                'data_augmentation': args.data_augmentation,
                'mlm_probability': args.mlm_probability if args.data_augmentation else None,
                
                # Hardware
                'device': str(device),
                'gpu_name': torch.cuda.get_device_name(args.gpu_id) if torch.cuda.is_available() else 'CPU',
            }
        )
        
        # Watch model (optional, can be memory intensive)
        # wandb.watch(model, log='all', log_freq=args.logging_steps)
        
        print(f"✓ Wandb initialized: {wandb.run.url}")
        print(f"✓ Wandb logging enabled - metrics will appear in Charts tab")
        print(f"  Logging every {args.logging_steps} steps")
    elif args.use_wandb and not WANDB_AVAILABLE:
        print("Warning: Wandb requested but not available. Install with: pip install wandb")
    
    # Debug: print wandb status
    if use_wandb:
        print(f"DEBUG: use_wandb = {use_wandb}, wandb.run = {wandb.run is not None}")
    
    # Save training arguments
    with open(os.path.join(output_dir, 'training_args.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Log directory: {log_dir}")
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")
    
    # Training loop
    global_step = 0
    best_perplexity = float('inf')
    model.train()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0.0
        
        progress_bar = tqdm(
            enumerate(train_dataloader), 
            total=len(train_dataloader),
            desc=f"Epoch {epoch + 1}"
        )
        
        for batch_idx, batch in progress_bar:
            if global_step >= num_training_steps:
                break
            
            inputs = batch['input_ids'].to(device, non_blocking=True)
            labels = batch['input_ids'].to(device, non_blocking=True)
            
            # Optional: Data augmentation with random masking
            if args.data_augmentation:
                mask = torch.rand(inputs.shape, device=device) < args.mlm_probability
                # Don't mask padding tokens
                mask = mask & (inputs != tokenizer.pad_token_id)
                # Replace masked tokens with random tokens or mask token
                inputs = inputs.clone()
                random_tokens = torch.randint(len(tokenizer), inputs.shape, device=device)
                inputs[mask] = random_tokens[mask]
            
            # Forward pass with mixed precision
            if use_amp:
                with autocast(device_type='cuda', dtype=amp_dtype):
                    outputs = model(inputs)
                    logits = outputs.logits
                    
                    # Shift for next-token prediction
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    loss = loss / args.accumulation_steps
                
                if scaler is not None:  # fp16 path
                    scaler.scale(loss).backward()
                else:  # bf16 path: no scaler needed
                    loss.backward()
            else:
                outputs = model(inputs)
                logits = outputs.logits
                
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss = loss / args.accumulation_steps
                
                loss.backward()
            
            # NaN/Inf detection for training stability
            current_loss_val = loss.item() * args.accumulation_steps
            if math.isnan(current_loss_val) or math.isinf(current_loss_val):
                print(f"\n⚠️ WARNING: NaN/Inf loss detected at batch {batch_idx}, step ~{global_step}!")
                print(f"  Loss value: {current_loss_val}")
                if use_amp:
                    print("  This may be caused by mixed precision overflow.")
                    print("  Consider: lowering LR, increasing warmup, or switching to bf16.")
                # Skip this batch - zero out gradients
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.update()
                continue
            
            epoch_loss += current_loss_val
            
            # Gradient accumulation
            if (batch_idx + 1) % args.accumulation_steps == 0:
                if scaler is not None:  # fp16 with GradScaler
                    scaler.unscale_(optimizer)
                    
                    # Check for inf/nan in gradients after unscaling
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    
                    # If grad_norm is inf/nan, scaler will skip this step
                    if math.isnan(grad_norm.item()) or math.isinf(grad_norm.item()):
                        print(f"\n⚠️ WARNING: Inf/NaN gradient norm at step {global_step}! "
                              f"Scaler will skip this optimizer step.")
                    
                    scaler.step(optimizer)
                    scaler.update()
                else:  # bf16 or fp32: no scaler
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                
                # Step scheduler AFTER optimizer step
                scheduler.step()
                optimizer.zero_grad()
                
                # Update EMA
                if ema is not None:
                    ema.update()
                
                # Increment global_step (this is the optimizer step counter)
                global_step += 1
                
                # Logging
                if global_step % args.logging_steps == 0:
                    # Get learning rates for all param groups
                    all_lrs = scheduler.get_last_lr()
                    # Display the last layer's LR (highest) instead of first layer
                    current_lr = all_lrs[-1] if len(all_lrs) > 0 else all_lrs[0]
                    train_loss = loss.item() * args.accumulation_steps
                    grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                    
                    # Determine current scheduler phase
                    phase = ""
                    if args.scheduler == 'wsd':
                        wsd_stable_end = num_warmup_steps + num_stable_steps
                        if global_step < num_warmup_steps:
                            phase = "warmup"
                        elif global_step < wsd_stable_end:
                            phase = "stable"
                        else:
                            phase = "decay"
                    elif global_step < num_warmup_steps:
                        phase = "warmup"
                    
                    # Tensorboard logging
                    writer.add_scalar('Train/Loss', train_loss, global_step)
                    writer.add_scalar('Train/LearningRate', current_lr, global_step)
                    writer.add_scalar('Train/GradNorm', grad_norm_val, global_step)
                    if scaler is not None:
                        writer.add_scalar('Train/ScalerScale', scaler.get_scale(), global_step)
                    
                    # Wandb logging
                    if use_wandb:
                        try:
                            log_dict = {
                                'train/loss': train_loss,
                                'train/learning_rate': current_lr,
                                'train/grad_norm': grad_norm_val,
                                'train/epoch': epoch + 1,
                                'train/step': global_step,
                            }
                            if scaler is not None:
                                log_dict['train/scaler_scale'] = scaler.get_scale()
                            if phase:
                                log_dict['train/lr_phase'] = phase
                            wandb.log(log_dict, step=global_step)
                            
                            # Debug print (first few times only)
                            if global_step <= 300:
                                phase_str = f" [{phase}]" if phase else ""
                                print(f"\n[Wandb] step {global_step}{phase_str}: "
                                      f"loss={train_loss:.4f}, lr={current_lr:.2e}, "
                                      f"grad_norm={grad_norm_val:.4f}")
                        except Exception as e:
                            print(f"\n[Wandb ERROR] Failed to log at step {global_step}: {e}")
                    
                    progress_bar.set_postfix({
                        'loss': f'{train_loss:.4f}',
                        'lr': f'{current_lr:.2e}',
                        'gn': f'{grad_norm_val:.2f}'
                    })
                
                # Evaluation
                if global_step % args.eval_steps == 0 and global_step > 0:
                    print(f"\nEvaluating at step {global_step}...")
                    
                    # Use EMA weights for evaluation if available
                    if ema is not None:
                        ema.apply_shadow()
                    
                    avg_loss, perplexity = evaluate_model(
                        model, test_dataloader, device, max_batches=args.eval_samples,
                        amp_dtype=amp_dtype if use_amp else None,
                    )
                    
                    # Restore original weights after evaluation
                    if ema is not None:
                        ema.restore()
                    
                    # Tensorboard logging
                    writer.add_scalar('Eval/Loss', avg_loss, global_step)
                    writer.add_scalar('Eval/Perplexity', perplexity, global_step)
                    
                    # Wandb logging
                    if use_wandb:
                        wandb.log({
                            'eval/loss': avg_loss,
                            'eval/perplexity': perplexity,
                            'eval/step': global_step,
                        }, step=global_step)
                    
                    print(f"Evaluation - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
                    
                    # Save best model (with EMA weights if enabled)
                    if perplexity < best_perplexity:
                        best_perplexity = perplexity
                        best_model_path = os.path.join(output_dir, 'best_model')
                        print(f"New best model! Saving to {best_model_path}")
                        
                        # Apply EMA weights before saving if available
                        if ema is not None:
                            ema.apply_shadow()
                        
                        model.save_pretrained(best_model_path)
                        tokenizer.save_pretrained(best_model_path)
                        
                        # Restore original weights
                        if ema is not None:
                            ema.restore()
                        
                        # Log best model to wandb
                        if use_wandb:
                            wandb.run.summary['best_perplexity'] = best_perplexity
                            wandb.run.summary['best_eval_loss'] = avg_loss
                            wandb.run.summary['best_step'] = global_step
                    
                    model.train()
                
                # Save checkpoint
                if global_step % args.save_steps == 0 and global_step > 0:
                    checkpoint_path = os.path.join(output_dir, f'checkpoint-{global_step}')
                    print(f"\nSaving checkpoint to {checkpoint_path}")
                    model.save_pretrained(checkpoint_path)
                    tokenizer.save_pretrained(checkpoint_path)
        
        # End of epoch
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"\nEpoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Evaluate at end of epoch
        print(f"End of epoch evaluation...")
        avg_loss, perplexity = evaluate_model(
            model, test_dataloader, device, max_batches=args.eval_samples,
            amp_dtype=amp_dtype if use_amp else None,
        )
        print(f"Evaluation - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
        
        # Tensorboard logging
        writer.add_scalar('Eval/EpochLoss', avg_loss, epoch)
        writer.add_scalar('Eval/EpochPerplexity', perplexity, epoch)
        
        # Wandb logging
        if use_wandb:
            wandb.log({
                'epoch/train_loss': avg_epoch_loss,
                'epoch/eval_loss': avg_loss,
                'epoch/eval_perplexity': perplexity,
                'epoch/number': epoch + 1,
            }, step=global_step)
    
    # Final save (with EMA weights if enabled)
    final_model_path = os.path.join(output_dir, 'final_model')
    print(f"\nTraining completed! Saving final model to {final_model_path}")
    
    # Apply EMA weights before final save if available
    if ema is not None:
        print("Applying EMA weights for final model...")
        ema.apply_shadow()
    
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    # Final evaluation (with EMA weights if they were applied)
    print("\nFinal evaluation on test set...")
    avg_loss, perplexity = evaluate_model(
        model, test_dataloader, device,
        amp_dtype=amp_dtype if use_amp else None,
    )
    print(f"Final Test Loss: {avg_loss:.4f}")
    print(f"Final Test Perplexity: {perplexity:.4f}")
    
    # Save final metrics
    with open(os.path.join(output_dir, 'final_metrics.txt'), 'w') as f:
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Final Test Loss: {avg_loss:.4f}\n")
        f.write(f"Final Test Perplexity: {perplexity:.4f}\n")
        f.write(f"Best Perplexity: {best_perplexity:.4f}\n")
        f.write(f"\nAdvanced Techniques:\n")
        f.write(f"  EMA: {args.use_ema} (decay={args.ema_decay})\n")
        f.write(f"  Label Smoothing: {args.label_smoothing}\n")
        f.write(f"  Layer-wise LR Decay: {args.layer_wise_lr_decay}\n")
        f.write(f"  Data Augmentation: {args.data_augmentation}\n")
        f.write(f"  Scheduler: {args.scheduler}\n")
    
    # Log final metrics to wandb
    if use_wandb:
        wandb.run.summary['final_test_loss'] = avg_loss
        wandb.run.summary['final_test_perplexity'] = perplexity
        wandb.run.summary['best_perplexity'] = best_perplexity
        wandb.run.summary['total_steps'] = global_step
        wandb.run.summary['output_dir'] = output_dir
        wandb.run.summary['model_type'] = model_type
        
        # Finish wandb run
        wandb.finish()
        print("✓ Wandb run completed")
    
    writer.close()
    print(f"\nAll outputs saved to: {output_dir}")
    
    return model, tokenizer


def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Print model type information
    print("\n" + "="*60)
    print("Mamba/Mamba2 Domain Continue Pre-training")
    print("="*60)
    print(f"\nAvailable Mamba1 models: {list(MAMBA1_MODELS.keys())}")
    print(f"Available Mamba2 models: {list(MAMBA2_MODELS.keys())}")
    print(f"\nSelected model: {args.model_name}")
    
    # Train model
    model, tokenizer = train_mamba2(args)
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print("="*50)


if __name__ == "__main__":
    main()
