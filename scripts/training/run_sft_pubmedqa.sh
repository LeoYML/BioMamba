#!/bin/bash

# Unified SFT launcher for PubMedQA
# Usage:
#   bash run_sft_pubmedqa.sh                          # default: full SFT
#   TRAINING_MODE=lora bash run_sft_pubmedqa.sh      # LoRA SFT
#   REPROCESS_DATA=true bash run_sft_pubmedqa.sh     # force re-tokenize dataset

cd "$(dirname "$0")/../.."

set -euo pipefail

# ============================================
# Configuration (can be overridden by env vars)
# ============================================

# Model settings
MODEL_NAME="${MODEL_NAME:-mamba2-130m}"
MODEL_PATH="${MODEL_PATH:-./checkpoints/biomamba2_mamba2-130m_20260209_002521/best_model}"

# Training mode: full | lora
TRAINING_MODE="${TRAINING_MODE:-full}"
if [[ "$TRAINING_MODE" != "full" && "$TRAINING_MODE" != "lora" ]]; then
  echo "ERROR: TRAINING_MODE must be 'full' or 'lora' (got: $TRAINING_MODE)"
  exit 1
fi

# LoRA settings (only used when TRAINING_MODE=lora)
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"

# Training settings
BATCH_SIZE="${BATCH_SIZE:-16}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-3e-5}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

# Logging/saving
LOGGING_STEPS="${LOGGING_STEPS:-10}"
EVAL_STEPS="${EVAL_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-20}"

# Data
REPROCESS_DATA="${REPROCESS_DATA:-false}"
NUM_PROC="${NUM_PROC:-1}"
MIX_BIOASQ="${MIX_BIOASQ:-false}"
BIOASQ_DATA_PATH="${BIOASQ_DATA_PATH:-}"
BIOASQ_SPLIT="${BIOASQ_SPLIT:-train}"
BIOASQ_TRAIN_RATIO="${BIOASQ_TRAIN_RATIO:-0.7}"
BIOASQ_MAX_TRAIN_SAMPLES="${BIOASQ_MAX_TRAIN_SAMPLES:-0}"

# Hardware
GPU_ID="${GPU_ID:-0}"
USE_FP16="${USE_FP16:-true}"
USE_BF16="${USE_BF16:-false}"
if [[ "$USE_BF16" == "true" ]]; then
  USE_FP16="false"
fi

# Output
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints}"
LOG_DIR="${LOG_DIR:-./runs}"

# Wandb (optional)
USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-biomamba2-sft}"

# ============================================
# Display configuration
# ============================================

echo "======================================================"
echo "Mamba2 SFT Training on PubMedQA"
echo "======================================================"
echo "Model: $MODEL_NAME"
echo "CPT Checkpoint: $MODEL_PATH"
echo "Training mode: $TRAINING_MODE"
if [[ "$TRAINING_MODE" == "lora" ]]; then
  echo "LoRA: rank=$LORA_RANK, alpha=$LORA_ALPHA"
else
  echo "LoRA: disabled (full fine-tuning)"
fi
echo "Batch size: $BATCH_SIZE x $ACCUMULATION_STEPS = $((BATCH_SIZE * ACCUMULATION_STEPS))"
echo "Learning rate: $LEARNING_RATE"
echo "Epochs: $NUM_EPOCHS"
echo "Max length: $MAX_LENGTH"
echo "Eval steps: $EVAL_STEPS"
echo "Save steps: $SAVE_STEPS"
echo "Precision: fp16=$USE_FP16 bf16=$USE_BF16"
echo "Reprocess data: $REPROCESS_DATA"
echo "Data num_proc: $NUM_PROC"
echo "Mix BioASQ: $MIX_BIOASQ"
if [[ "$MIX_BIOASQ" == "true" ]]; then
  echo "BioASQ data path: $BIOASQ_DATA_PATH"
  echo "BioASQ split: $BIOASQ_SPLIT"
  echo "BioASQ train ratio: $BIOASQ_TRAIN_RATIO"
  echo "BioASQ max train samples: $BIOASQ_MAX_TRAIN_SAMPLES"
fi
echo "======================================================"
echo ""

# Validate checkpoint path
if [ ! -d "$MODEL_PATH" ]; then
  echo "ERROR: Model path does not exist: $MODEL_PATH"
  exit 1
fi

# Build command
CMD="python scripts/training/finetune_pubmedqa_sft.py \
  --model_name $MODEL_NAME \
  --model_path $MODEL_PATH \
  --batch_size $BATCH_SIZE \
  --accumulation_steps $ACCUMULATION_STEPS \
  --lr $LEARNING_RATE \
  --num_epochs $NUM_EPOCHS \
  --max_length $MAX_LENGTH \
  --warmup_ratio $WARMUP_RATIO \
  --weight_decay $WEIGHT_DECAY \
  --max_grad_norm $MAX_GRAD_NORM \
  --output_dir $OUTPUT_DIR \
  --log_dir $LOG_DIR \
  --gpu_id $GPU_ID \
  --num_proc $NUM_PROC \
  --logging_steps $LOGGING_STEPS \
  --eval_steps $EVAL_STEPS \
  --save_steps $SAVE_STEPS"

if [[ "$USE_FP16" == "true" ]]; then
  CMD="$CMD --fp16"
fi
if [[ "$USE_BF16" == "true" ]]; then
  CMD="$CMD --bf16"
fi

if [[ "$TRAINING_MODE" == "lora" ]]; then
  CMD="$CMD --use_lora --lora_rank $LORA_RANK --lora_alpha $LORA_ALPHA"
fi

if [[ "$REPROCESS_DATA" == "true" ]]; then
  CMD="$CMD --reprocess_data"
fi

if [[ "$MIX_BIOASQ" == "true" ]]; then
  if [[ -z "$BIOASQ_DATA_PATH" ]]; then
    echo "ERROR: MIX_BIOASQ=true but BIOASQ_DATA_PATH is empty"
    exit 1
  fi
  CMD="$CMD --mix_bioasq \
    --bioasq_data_path $BIOASQ_DATA_PATH \
    --bioasq_split $BIOASQ_SPLIT \
    --bioasq_train_ratio $BIOASQ_TRAIN_RATIO"

  if [[ "$BIOASQ_MAX_TRAIN_SAMPLES" != "0" ]]; then
    CMD="$CMD --bioasq_max_train_samples $BIOASQ_MAX_TRAIN_SAMPLES"
  fi
fi

if [[ "$USE_WANDB" == "true" ]]; then
  CMD="$CMD --use_wandb --wandb_project $WANDB_PROJECT"
fi

# Prevent OpenMP oversubscription in constrained containers
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Print memory info
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU Memory Info:"
  nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
  echo ""
fi

echo "Running command:"
echo "$CMD"
echo ""

eval "$CMD"

echo ""
echo "======================================================"
echo "Training completed!"
echo "======================================================"
