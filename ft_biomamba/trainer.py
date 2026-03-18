"""
Unified Trainer for both CPT and SFT stages.

Key features:
- WSD scheduler for CPT, cosine for SFT
- EMA, label smoothing, layer-wise LR decay
- BF16 mixed precision
- Gradient checkpointing
- Best-model tracking with early stopping patience
- Multi-GPU DDP support with no_sync gradient accumulation
"""

import os
import math
import json
from contextlib import nullcontext
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False

from .config import PROJECT_ROOT
from .model import EMA, LabelSmoothingLoss, get_layer_lr_groups


# ---------------------------------------------------------------------------
# WSD (Warmup-Stable-Decay) scheduler
# ---------------------------------------------------------------------------
def get_wsd_schedule(optimizer, num_warmup, num_stable, num_decay, min_lr_ratio=0.1):
    def _lr_lambda(step):
        if step < num_warmup:
            return float(step) / float(max(1, num_warmup))
        elif step < num_warmup + num_stable:
            return 1.0
        else:
            d = step - num_warmup - num_stable
            progress = float(d) / float(max(1, num_decay))
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, _lr_lambda)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Trainer:
    """Unified trainer for CPT and SFT stages."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        train_dataset,
        eval_dataset,
        *,
        # training mode
        mode: str = "cpt",  # "cpt" or "sft"
        # hyperparams
        batch_size: int = 16,
        accumulation_steps: int = 8,
        lr: float = 5e-5,
        weight_decay: float = 0.1,
        num_epochs: int = 3,
        warmup_ratio: float = 0.1,
        stable_ratio: float = 0.65,
        decay_ratio: float = 0.20,
        min_lr_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        scheduler_type: str = "wsd",
        # advanced
        use_ema: bool = False,
        ema_decay: float = 0.999,
        label_smoothing: float = 0.0,
        layer_lr_decay: float = 1.0,
        # precision
        bf16: bool = True,
        # logging / saving
        output_dir: str = os.path.join(PROJECT_ROOT, "checkpoints"),
        log_dir: str = os.path.join(PROJECT_ROOT, "runs"),
        logging_steps: int = 50,
        eval_steps: int = 500,
        save_steps: int = 1000,
        run_name: Optional[str] = None,
        # trainable params override (for LoRA)
        trainable_params=None,
        # hardware
        gpu_id: int = 0,
        # early stopping
        patience: int = 5,
    ):
        self.tokenizer = tokenizer
        self.mode = mode

        # Distributed setup
        self.distributed = torch.distributed.is_initialized()
        if self.distributed:
            self.rank = torch.distributed.get_rank()
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.world_size = torch.distributed.get_world_size()
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        else:
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
            if torch.cuda.is_available():
                self.device = torch.device("cuda", gpu_id)
                torch.cuda.set_device(gpu_id)
            else:
                self.device = torch.device("cpu")

        self.is_main = (self.rank == 0)
        model.to(self.device)

        # Wrap model with DDP
        if self.distributed:
            self.model = DDP(model, device_ids=[self.local_rank],
                             output_device=self.local_rank,
                             find_unused_parameters=False)
            self._raw_model = model  # unwrapped model for save/EMA
        else:
            self.model = model
            self._raw_model = model

        # Data — use DistributedSampler when running multi-GPU
        num_workers = min(8, os.cpu_count() or 1)
        pin = self.device.type == "cuda"
        self.train_sampler = DistributedSampler(
            train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True
        ) if self.distributed else None
        self.train_loader = DataLoader(
            train_dataset,
            shuffle=(self.train_sampler is None),
            sampler=self.train_sampler,
            batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=self.distributed,  # avoid uneven batches in DDP
        )
        self.eval_loader = DataLoader(
            eval_dataset, shuffle=False, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )

        # Optimizer (use raw model for param groups)
        if trainable_params is not None:
            self._trainable = trainable_params
            self.optimizer = AdamW(trainable_params, lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
        elif layer_lr_decay < 1.0:
            groups = get_layer_lr_groups(self._raw_model, lr, weight_decay, layer_lr_decay)
            self._trainable = [p for g in groups for p in g["params"]]
            self.optimizer = AdamW(groups, betas=(0.9, 0.95))
        else:
            self._trainable = [p for p in self._raw_model.parameters() if p.requires_grad]
            self.optimizer = AdamW(self._trainable, lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)

        # Scheduler
        self.accumulation_steps = accumulation_steps
        steps_per_epoch = math.ceil(len(self.train_loader) / accumulation_steps)
        self.total_steps = steps_per_epoch * num_epochs
        num_warmup = int(warmup_ratio * self.total_steps)

        if scheduler_type == "wsd":
            num_stable = int(stable_ratio * self.total_steps)
            num_decay_steps = self.total_steps - num_warmup - num_stable
            self.scheduler = get_wsd_schedule(self.optimizer, num_warmup, num_stable, num_decay_steps, min_lr_ratio)
        elif scheduler_type == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup, self.total_steps)
        else:
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup, self.total_steps)

        # AMP
        self.bf16 = bf16 and self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if self.bf16 else None

        # EMA (on raw model)
        self.ema = EMA(self._raw_model, ema_decay) if use_ema else None

        # Loss
        if label_smoothing > 0 and mode == "cpt":
            self.loss_fn = LabelSmoothingLoss(label_smoothing)
        else:
            self.loss_fn = None  # use F.cross_entropy directly

        # Bookkeeping
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.patience = patience

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name or f"biomamba_{mode}_{ts}"
        self.output_dir = os.path.join(output_dir, self.run_name)
        if self.is_main:
            os.makedirs(self.output_dir, exist_ok=True)

        self.writer = (
            SummaryWriter(os.path.join(log_dir, self.run_name))
            if TB_AVAILABLE and self.is_main else None
        )

        # Print summary (rank 0 only)
        if self.is_main:
            total_p = sum(p.numel() for p in self._raw_model.parameters())
            train_p = sum(p.numel() for p in self._trainable)
            eff_batch = batch_size * accumulation_steps * self.world_size
            print(f"\n{'='*60}")
            print(f"Trainer [{mode.upper()}]  device={self.device}  world_size={self.world_size}")
            print(f"  Total params: {total_p:,}  Trainable: {train_p:,} ({100*train_p/total_p:.1f}%)")
            print(f"  Batch/GPU: {batch_size}  Accum: {accumulation_steps}  "
                  f"GPUs: {self.world_size}  Eff: {eff_batch}")
            print(f"  LR: {lr}  Epochs: {num_epochs}  Steps: {self.total_steps}")
            print(f"  EMA: {use_ema}  LabelSmooth: {label_smoothing}  LayerDecay: {layer_lr_decay}")
            print(f"  Output: {self.output_dir}")
            print(f"{'='*60}\n")

    # -------------------------------------------------------------------
    # Loss computation
    # -------------------------------------------------------------------
    def _compute_loss(self, inputs, labels):
        outputs = self.model(inputs)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        if self.loss_fn is not None:
            loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        else:
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return loss

    # -------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, max_batches: int = 100) -> tuple:
        self.model.eval()
        if self.ema:
            self.ema.apply(self._raw_model)

        total_loss = 0.0
        n = 0
        for i, batch in enumerate(self.eval_loader):
            if i >= max_batches:
                break
            inputs = batch["input_ids"].to(self.device, non_blocking=True)
            if self.mode == "sft":
                labels = batch["labels"].to(self.device, non_blocking=True)
            else:
                labels = inputs.clone()
                if "attention_mask" in batch:
                    mask = batch["attention_mask"].to(self.device, non_blocking=True)
                    labels[mask == 0] = -100

            if self.amp_dtype:
                with autocast(device_type="cuda", dtype=self.amp_dtype):
                    loss = self._compute_loss(inputs, labels)
            else:
                loss = self._compute_loss(inputs, labels)

            if torch.isfinite(loss):
                total_loss += loss.item()
                n += 1

        if self.ema:
            self.ema.restore(self._raw_model)
        self.model.train()

        avg = total_loss / n if n > 0 else 0.0
        ppl = math.exp(avg) if avg < 20 else float("inf")
        return avg, ppl

    # -------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------
    def _save(self, path):
        if not self.is_main:
            return
        os.makedirs(path, exist_ok=True)
        if self.ema:
            self.ema.apply(self._raw_model)
        self._raw_model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        if self.ema:
            self.ema.restore(self._raw_model)

    # -------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------
    def train(self) -> str:
        """Run training and return path to best model."""
        best_loss = float("inf")
        best_step = -1
        stale = 0
        global_step = 0
        self.model.train()
        self.optimizer.zero_grad()

        for epoch in range(self.num_epochs):
            # Set epoch for DistributedSampler to ensure proper shuffling
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            if self.is_main:
                print(f"\n--- Epoch {epoch+1}/{self.num_epochs} ---")
            epoch_loss = 0.0

            if self.is_main:
                pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                            desc=f"Epoch {epoch+1}")
            else:
                pbar = enumerate(self.train_loader)

            for batch_idx, batch in pbar:
                inputs = batch["input_ids"].to(self.device, non_blocking=True)
                if self.mode == "sft":
                    labels = batch["labels"].to(self.device, non_blocking=True)
                else:
                    labels = inputs.clone()
                    # Mask padding tokens so loss ignores them
                    if "attention_mask" in batch:
                        mask = batch["attention_mask"].to(self.device, non_blocking=True)
                        labels[mask == 0] = -100

                # Determine if this is an accumulation step (skip allreduce)
                should_step = (
                    (batch_idx + 1) % self.accumulation_steps == 0
                    or (batch_idx + 1) == len(self.train_loader)
                )

                # Use no_sync to skip allreduce on accumulation-only steps
                if not should_step and self.distributed:
                    sync_ctx = self.model.no_sync()
                else:
                    sync_ctx = nullcontext()

                with sync_ctx:
                    # Forward
                    if self.amp_dtype:
                        with autocast(device_type="cuda", dtype=self.amp_dtype):
                            loss = self._compute_loss(inputs, labels)
                    else:
                        loss = self._compute_loss(inputs, labels)

                    if not torch.isfinite(loss):
                        continue

                    loss_scaled = loss / self.accumulation_steps
                    loss_scaled.backward()

                epoch_loss += loss.item()

                # Step
                if not should_step:
                    continue

                torch.nn.utils.clip_grad_norm_(self._trainable, self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                global_step += 1

                if self.ema:
                    self.ema.update(self._raw_model)

                # Logging (rank 0 only)
                if self.is_main and global_step % self.logging_steps == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")
                    if self.writer:
                        self.writer.add_scalar("train/loss", loss.item(), global_step)
                        self.writer.add_scalar("train/lr", lr, global_step)

                # Eval (rank 0 only, other ranks wait at barrier)
                if global_step % self.eval_steps == 0:
                    if self.is_main:
                        val_loss, val_ppl = self.evaluate()
                        print(f"  [step {global_step}] val_loss={val_loss:.4f}  ppl={val_ppl:.2f}")
                        if self.writer:
                            self.writer.add_scalar("eval/loss", val_loss, global_step)
                            self.writer.add_scalar("eval/ppl", val_ppl, global_step)

                        if val_loss < best_loss:
                            best_loss = val_loss
                            best_step = global_step
                            stale = 0
                            self._save(os.path.join(self.output_dir, "best_model"))
                            print(f"  >> New best model saved (loss={best_loss:.4f})")
                        else:
                            stale += 1
                            if stale >= self.patience:
                                print(f"  Early stopping at step {global_step} (patience={self.patience})")

                    # Broadcast early stopping decision from rank 0
                    if self.distributed:
                        stop_tensor = torch.tensor([1 if stale >= self.patience else 0],
                                                   device=self.device)
                        torch.distributed.broadcast(stop_tensor, src=0)
                        if stop_tensor.item():
                            break
                        torch.distributed.barrier()
                    elif stale >= self.patience:
                        break

                # Periodic save
                if global_step % self.save_steps == 0:
                    self._save(os.path.join(self.output_dir, f"checkpoint-{global_step}"))
                    if self.distributed:
                        torch.distributed.barrier()

            if stale >= self.patience:
                break

            # End-of-epoch eval
            if self.is_main:
                val_loss, val_ppl = self.evaluate()
                avg_train = epoch_loss / len(self.train_loader)
                print(f"  Epoch {epoch+1}: train_loss={avg_train:.4f}  val_loss={val_loss:.4f}  ppl={val_ppl:.2f}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_step = global_step
                    stale = 0
                    self._save(os.path.join(self.output_dir, "best_model"))
            if self.distributed:
                torch.distributed.barrier()

        # Final save
        self._save(os.path.join(self.output_dir, "final_model"))
        if self.is_main:
            metrics = {
                "best_val_loss": best_loss,
                "best_step": best_step,
                "total_steps": global_step,
            }
            with open(os.path.join(self.output_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)

        if self.writer:
            self.writer.close()

        best_path = os.path.join(self.output_dir, "best_model")
        if self.is_main:
            print(f"\nTraining complete. Best model: {best_path} (loss={best_loss:.4f} @ step {best_step})")
        return best_path
