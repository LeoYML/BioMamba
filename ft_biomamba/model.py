"""
Model utilities: loading, LoRA injection, EMA, parameter counting.
"""

import math
import copy
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel

from .config import MAMBA2_MODELS, TOKENIZER_NAME


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        vocab = tokenizer.get_vocab()
        if "<|padding|>" in vocab:
            tokenizer.pad_token = "<|padding|>"
        else:
            tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(name_or_path: str, device: str = "cuda", gradient_checkpointing: bool = False) -> MambaLMHeadModel:
    """Load a Mamba2 model from HF hub name or local checkpoint path."""
    # Resolve short name -> HF path
    if name_or_path in MAMBA2_MODELS:
        name_or_path = MAMBA2_MODELS[name_or_path]["path"]
    model = MambaLMHeadModel.from_pretrained(name_or_path, device=device)
    if gradient_checkpointing:
        enable_gradient_checkpointing(model)
    return model


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ---------------------------------------------------------------------------
# Gradient checkpointing
# ---------------------------------------------------------------------------
def enable_gradient_checkpointing(model: MambaLMHeadModel):
    """Wrap each backbone layer with gradient checkpointing to trade compute for memory."""
    for i, layer in enumerate(model.backbone.layers):
        orig_forward = layer.forward

        def _make_ckpt_fwd(fn):
            def _fwd(*args, **kwargs):
                # torch.utils.checkpoint requires at least one tensor with requires_grad
                return checkpoint(fn, *args, use_reentrant=False, **kwargs)
            return _fwd

        layer.forward = _make_ckpt_fwd(orig_forward)
    print(f"  Gradient checkpointing enabled for {len(model.backbone.layers)} layers")


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """Low-rank adapter for a linear layer."""

    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.lora_A @ self.lora_B) * self.scaling


def inject_lora(model, rank: int = 16, alpha: int = 32,
                target_modules=("mixer.in_proj", "mixer.out_proj")):
    """Inject LoRA adapters into matching Linear layers and freeze the rest."""
    lora_params = []

    for name, module in model.named_modules():
        if not any(t in name for t in target_modules):
            continue
        if not isinstance(module, nn.Linear):
            continue

        lora = LoRALinear(module.in_features, module.out_features, rank, alpha)
        lora = lora.to(device=module.weight.device, dtype=module.weight.dtype)
        setattr(module, "lora_layer", lora)

        # Freeze original weights
        module.weight.requires_grad = False
        if module.bias is not None:
            module.bias.requires_grad = False

        lora_params.extend([lora.lora_A, lora.lora_B])

        # Patch forward
        orig_forward = module.forward

        def _make_fwd(orig_fn):
            def _fwd(self, x):
                out = orig_fn(x)
                if hasattr(self, "lora_layer"):
                    out = out + self.lora_layer(x)
                return out
            return _fwd

        module.forward = _make_fwd(orig_forward).__get__(module, nn.Linear)

    return lora_params


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------
class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: OrderedDict = OrderedDict()
        self.backup: OrderedDict = OrderedDict()
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply(self, model: nn.Module):
        """Replace model params with EMA shadow (for eval)."""
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original params after eval."""
        for name, p in model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup.clear()


# ---------------------------------------------------------------------------
# Layer-wise LR decay
# ---------------------------------------------------------------------------
def get_layer_lr_groups(model, lr: float, weight_decay: float, layer_decay: float = 0.85):
    """Create optimizer param groups with layer-wise LR decay."""
    layer_params = {}
    other_params = []
    num_layers = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer_num = None
        if "layers." in name:
            parts = name.split(".")
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    try:
                        layer_num = int(parts[i + 1])
                        num_layers = max(num_layers, layer_num + 1)
                    except ValueError:
                        pass
                    break

        if layer_num is not None:
            layer_params.setdefault(layer_num, []).append(param)
        else:
            other_params.append(param)

    groups = []
    if num_layers > 0:
        for ln in sorted(layer_params):
            layer_lr = lr * (layer_decay ** (num_layers - ln - 1))
            groups.append({"params": layer_params[ln], "lr": layer_lr, "weight_decay": weight_decay})
    if other_params:
        groups.append({"params": other_params, "lr": lr, "weight_decay": weight_decay})

    if not groups:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                   "lr": lr, "weight_decay": weight_decay}]
    return groups


# ---------------------------------------------------------------------------
# Label-smoothing loss
# ---------------------------------------------------------------------------
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing: float = 0.1, ignore_index: int = -100):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (N, V), targets: (N,)
        mask = targets != self.ignore_index
        if not mask.any():
            return logits.sum() * 0.0

        logits = logits[mask]
        targets = targets[mask]

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        nll = torch.nn.functional.nll_loss(log_probs, targets, reduction="mean")
        smooth = -log_probs.mean(dim=-1).mean()
        return (1.0 - self.smoothing) * nll + self.smoothing * smooth
