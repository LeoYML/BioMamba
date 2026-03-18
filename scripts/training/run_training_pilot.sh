#!/bin/bash

################################################################################
# Pilot / Small-budget Experiment Script
# 
# Purpose: Run a short training to validate settings before committing to 
# full training. Useful for:
#   - Checking if LR is appropriate (loss should decrease, not explode)
#   - Verifying no NaN/Inf issues with mixed precision
#   - Comparing different LR / scheduler combinations
#   - Drawing early loss curves in wandb to see "dip then recover" patterns
#
# Typical usage:
#   bash run_training_pilot.sh                    # Default: 500 steps, wsd
#   bash run_training_pilot.sh --steps 1000       # Custom step count
#   bash run_training_pilot.sh --lr 1e-5          # Test different LR
#   bash run_training_pilot.sh --scheduler cosine # Compare with cosine
################################################################################

cd "$(dirname "$0")/../.."

set -e
source ./.venv/bin/activate

################################################################################
# Default Configuration (designed for quick experiments)
################################################################################

MODEL_NAME="mamba2-130m"

# Short training - just enough to see if the curve is healthy
NUM_TRAINING_STEPS=500     # Override epochs with fixed steps
LEARNING_RATE=1e-5         # Slightly aggressive to test gradient response
BATCH_SIZE=16
ACCUMULATION_STEPS=16
MAX_LENGTH=1024
WARMUP_RATIO=0.15          # 15% warmup (longer to suppress initial grad spikes)
MAX_GRAD_NORM=1.0

# Default: WSD scheduler
SCHEDULER="wsd"
STABLE_RATIO=0.7
DECAY_RATIO=0.2
MIN_LR_RATIO=0.1

# Advanced techniques
USE_EMA=false              # Disable for pilot (saves memory/time)
EMA_DECAY=0.999
LABEL_SMOOTHING=0.0        # Disable for pilot
LAYER_LR_DECAY=1.0         # Disable for pilot (cleaner comparison)
DATA_AUGMENTATION=false

# Performance: use bf16 on H100/A100 (faster, more stable than fp16)
USE_BF16=true

# Frequent logging for detailed curves
LOGGING_STEPS=10           # Every 10 steps for detailed curves
EVAL_STEPS=100             # Eval every 100 steps (was 50, reduced overhead)
SAVE_STEPS=500             # Only save at the very end in pilot

USE_WANDB=true
WANDB_PROJECT="biomamba-pilot"

GPU_ID=0

################################################################################
# Parse command-line arguments (override defaults)
################################################################################

while [[ $# -gt 0 ]]; do
    case $1 in
        --steps)
            NUM_TRAINING_STEPS="$2"; shift 2 ;;
        --lr)
            LEARNING_RATE="$2"; shift 2 ;;
        --scheduler)
            SCHEDULER="$2"; shift 2 ;;
        --model)
            MODEL_NAME="$2"; shift 2 ;;
        --batch_size)
            BATCH_SIZE="$2"; shift 2 ;;
        --accumulation)
            ACCUMULATION_STEPS="$2"; shift 2 ;;
        --warmup)
            WARMUP_RATIO="$2"; shift 2 ;;
        --grad_clip)
            MAX_GRAD_NORM="$2"; shift 2 ;;
        --stable_ratio)
            STABLE_RATIO="$2"; shift 2 ;;
        --decay_ratio)
            DECAY_RATIO="$2"; shift 2 ;;
        --min_lr)
            MIN_LR_RATIO="$2"; shift 2 ;;
        --label_smoothing)
            LABEL_SMOOTHING="$2"; shift 2 ;;
        --layer_decay)
            LAYER_LR_DECAY="$2"; shift 2 ;;
        --no_wandb)
            USE_WANDB=false; shift ;;
        --fp16)
            USE_BF16=false; shift ;;
        --no_bf16)
            USE_BF16=false; shift ;;
        --gpu)
            GPU_ID="$2"; shift 2 ;;
        --run_name)
            WANDB_RUN_NAME="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

################################################################################
# Display
################################################################################

echo ""
echo "=============================================="
echo "🔬 PILOT EXPERIMENT"
echo "=============================================="
echo ""
echo "Model:           $MODEL_NAME"
echo "Total Steps:     $NUM_TRAINING_STEPS (fixed, not epoch-based)"
echo "Learning Rate:   $LEARNING_RATE"
echo "Batch Size:      $BATCH_SIZE"
echo "Accumulation:    $ACCUMULATION_STEPS"
echo "Effective Batch: $((BATCH_SIZE * ACCUMULATION_STEPS))"
echo "Warmup Ratio:    $WARMUP_RATIO"
echo "Grad Clip:       $MAX_GRAD_NORM"
echo "Scheduler:       $SCHEDULER"
if [ "$SCHEDULER" = "wsd" ]; then
echo "  Stable Ratio:  $STABLE_RATIO"
echo "  Decay Ratio:   $DECAY_RATIO"
echo "  Min LR Ratio:  $MIN_LR_RATIO"
fi
echo ""
echo "Logging:         every $LOGGING_STEPS steps"
echo "Eval:            every $EVAL_STEPS steps"
echo "bf16:            $USE_BF16"
echo "Wandb Project:   ${USE_WANDB:+$WANDB_PROJECT}"
echo ""
echo "=============================================="
echo ""

################################################################################
# GPU check
################################################################################

if command -v nvidia-smi &> /dev/null; then
    echo "GPU:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | head -n 1
    echo ""
fi

################################################################################
# Build command
################################################################################

CMD="python scripts/training/finetune_pubmed_medline.py \
    --model_name $MODEL_NAME \
    --num_training_steps $NUM_TRAINING_STEPS \
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
    CMD="$CMD --num_restarts ${NUM_RESTARTS:-3}"
fi

# Add bf16 (recommended for H100/A100)
if [ "$USE_BF16" = true ]; then
    CMD="$CMD --bf16"
fi

# Add wandb
if [ "$USE_WANDB" = true ]; then
    CMD="$CMD --use_wandb --wandb_project $WANDB_PROJECT"
    if [ -n "$WANDB_RUN_NAME" ]; then
        CMD="$CMD --wandb_run_name $WANDB_RUN_NAME"
    fi
fi

################################################################################
# Run
################################################################################

echo "Command: $CMD"
echo ""
echo "Starting pilot experiment..."
echo ""

eval $CMD

echo ""
echo "=============================================="
echo "Pilot experiment finished!"
echo "Check wandb ($WANDB_PROJECT) for loss curves."
echo ""
echo "What to look for:"
echo "  ✓ Loss should decrease over time"
echo "  ✓ No NaN/Inf warnings"  
echo "  ✓ Grad norm should be stable (not spiking)"
echo "  ✓ LR curve should show warmup → stable → decay"
echo ""
echo "If loss explodes: try --lr 1e-6"
echo "If loss barely moves: try --lr 1e-5"
echo "=============================================="
