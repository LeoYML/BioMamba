#!/bin/bash

################################################################################
# Advanced Domain Continue Pre-training Script
# Supports both Mamba1 and Mamba2 with optimization techniques
################################################################################

cd "$(dirname "$0")/../.."

set -e  # Exit on error
source ./.venv/bin/activate
# wandb login (use WANDB_API_KEY env var or `wandb login` interactively)

################################################################################
# Configuration
# 
# Settings optimized based on pilot experiments:
#   - LR=3e-5 gave best eval PPL (2.21 @ 200 steps)
#   - 15% warmup eliminates initial gradient spikes (was 42→0.4)
#   - grad_norm stable at ~0.12, well below clip=1.0
#   - EMA + label_smoothing help smooth noisy loss curves
################################################################################

# Choose your model (Mamba1 or Mamba2)
# Mamba1 options: mamba-130m, mamba-370m, mamba-790m, mamba-1.4b, mamba-2.8b
# Mamba2 options: mamba2-130m, mamba2-370m, mamba2-780m, mamba2-1.3b, mamba2-2.7b

MODEL_NAME="mamba2-130m"  # Change to "mamba-130m" for Mamba1

# Training configuration
# NOTE: LR=3e-5 was found optimal in pilot experiments (eval PPL: 2.21 @ 200 steps)
NUM_EPOCHS=3
LEARNING_RATE=3e-5         # Optimal from pilot sweep (tested: 5e-6, 1e-5, 3e-5)
BATCH_SIZE=40
ACCUMULATION_STEPS=16      # Effective batch = 16 * 16 = 256
MAX_LENGTH=1024
WARMUP_RATIO=0.15          # 15% warmup (longer warmup eliminates initial grad spikes)
MAX_GRAD_NORM=1.0          # Gradient clipping threshold

# LR Scheduler: wsd (warmup-stable-decay) recommended for continued pretraining
# Options: linear, cosine, cosine_restarts, wsd
SCHEDULER="wsd"

# WSD scheduler parameters (only used if SCHEDULER=wsd)
# Schedule: warmup(15%) → stable(65%) → cosine_decay(20%) of remaining steps
STABLE_RATIO=0.65          # 65% of post-warmup steps at constant LR
DECAY_RATIO=0.20           # 20% of post-warmup steps for cosine decay
MIN_LR_RATIO=0.1           # Decay to 10% of peak LR (3e-6 when peak=3e-5)

# Cosine restarts (only used if SCHEDULER=cosine_restarts)
NUM_RESTARTS=3

# Advanced optimization techniques
USE_EMA=true               # Exponential Moving Average
EMA_DECAY=0.999
LABEL_SMOOTHING=0.1        # 0.0 to disable, 0.1 recommended
LAYER_LR_DECAY=0.85        # 1.0 to disable, 0.75-0.9 recommended
DATA_AUGMENTATION=false    # MLM-style data augmentation (disabled by default)
MLM_PROBABILITY=0.15

# Logging - adjusted for full training (less frequent than pilot)
USE_WANDB=true
WANDB_PROJECT="biomamba-training"
LOGGING_STEPS=50          # Log every 100 optimizer steps
EVAL_STEPS=50             # Eval every 500 optimizer steps (~300 evals for 3 epochs)
SAVE_STEPS=300            # Save checkpoint every 1000 optimizer steps

# Performance acceleration
USE_BF16=true              # bf16 (recommended for H100/A100, faster than fp16, no scaler needed)
USE_COMPILE=false          # torch.compile (10-30% speedup, but first run takes 1-2min to compile)

# Hardware
GPU_ID=0

################################################################################
# Display configuration
################################################################################

echo ""
echo "=============================================="
echo "Advanced Domain Continue Pre-training"
echo "=============================================="
echo ""
echo "Model Configuration:"
echo "  Model:             $MODEL_NAME"
echo "  Max Length:        $MAX_LENGTH"
echo ""
echo "Training Configuration:"
echo "  Epochs:            $NUM_EPOCHS"
echo "  Learning Rate:     $LEARNING_RATE"
echo "  Batch Size:        ${BATCH_SIZE:-Auto}"
echo "  Accumulation:      $ACCUMULATION_STEPS"
echo "  Warmup Ratio:      $WARMUP_RATIO"
echo "  Max Grad Norm:     $MAX_GRAD_NORM"
echo "  Scheduler:         $SCHEDULER"
if [ "$SCHEDULER" = "wsd" ]; then
echo "  WSD Config:"
echo "    Stable Ratio:    $STABLE_RATIO"
echo "    Decay Ratio:     $DECAY_RATIO"
echo "    Min LR Ratio:    $MIN_LR_RATIO"
fi
echo ""
echo "Advanced Techniques:"
echo "  EMA:               $USE_EMA (decay=$EMA_DECAY)"
echo "  Label Smoothing:   $LABEL_SMOOTHING"
echo "  Layer LR Decay:    $LAYER_LR_DECAY"
echo "  Data Augmentation: $DATA_AUGMENTATION"
echo ""
echo "Acceleration:"
echo "  bf16:              $USE_BF16"
echo "  torch.compile:     $USE_COMPILE"
echo ""
echo "Monitoring:"
echo "  Logging Steps:     $LOGGING_STEPS"
echo "  Eval Steps:        $EVAL_STEPS"
echo "  Save Steps:        $SAVE_STEPS"
echo ""
echo "=============================================="
echo ""

################################################################################
# Check GPU
################################################################################

if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | head -n 1
    echo ""
fi

################################################################################
# Build command
################################################################################

CMD="python scripts/training/finetune_pubmed_medline.py \
    --model_name $MODEL_NAME \
    --num_epochs $NUM_EPOCHS \
    --lr $LEARNING_RATE \
    --accumulation_steps $ACCUMULATION_STEPS \
    --max_length $MAX_LENGTH \
    --warmup_ratio $WARMUP_RATIO \
    --max_grad_norm $MAX_GRAD_NORM \
    --scheduler $SCHEDULER \
    --label_smoothing $LABEL_SMOOTHING \
    --layer_wise_lr_decay $LAYER_LR_DECAY \
    --logging_steps $LOGGING_STEPS \
    --eval_steps $EVAL_STEPS \
    --save_steps $SAVE_STEPS \
    --gpu_id $GPU_ID"

# Add optional batch size
if [ -n "$BATCH_SIZE" ]; then
    CMD="$CMD --batch_size $BATCH_SIZE"
fi

# Add EMA
if [ "$USE_EMA" = true ]; then
    CMD="$CMD --use_ema --ema_decay $EMA_DECAY"
fi

# Add data augmentation
if [ "$DATA_AUGMENTATION" = true ]; then
    CMD="$CMD --data_augmentation --mlm_probability $MLM_PROBABILITY"
fi

# Add WSD scheduler parameters
if [ "$SCHEDULER" = "wsd" ]; then
    CMD="$CMD --stable_ratio $STABLE_RATIO --decay_ratio $DECAY_RATIO --min_lr_ratio $MIN_LR_RATIO"
fi

# Add cosine restarts
if [ "$SCHEDULER" = "cosine_restarts" ]; then
    CMD="$CMD --num_restarts $NUM_RESTARTS"
fi

# Add bf16 / fp16
if [ "$USE_BF16" = true ]; then
    CMD="$CMD --bf16"
fi

# Add torch.compile
if [ "$USE_COMPILE" = true ]; then
    CMD="$CMD --compile"
fi

# Add wandb
if [ "$USE_WANDB" = true ]; then
    CMD="$CMD --use_wandb --wandb_project $WANDB_PROJECT"
fi

################################################################################
# Run training
################################################################################

echo "Starting training..."
echo "Command: $CMD"
echo ""

eval $CMD

echo ""
echo "=============================================="
echo "Training completed!"
echo "=============================================="
