#!/bin/bash

# Mamba2 Domain Post-Training on PubMed-MEDLINE
# Configuration for larger model (2.7B)

# Configuration
cd "$(dirname "$0")/../.."

MODEL_NAME="mamba2-130m"
GPU_ID=0
NUM_EPOCHS=2
BATCH_SIZE=8  # Smaller batch size for large model
ACCUMULATION_STEPS=8 # Larger accumulation to maintain effective batch size
LEARNING_RATE=2e-5
MAX_LENGTH=2048

# Directories
DATA_DIR="./data"
OUTPUT_DIR="./checkpoints"
LOG_DIR="./runs"

# Training arguments
WARMUP_RATIO=0.1
WEIGHT_DECAY=0.1
MAX_GRAD_NORM=1.0
SCHEDULER="cosine"

# Logging and evaluation
LOGGING_STEPS=50
EVAL_STEPS=250
SAVE_STEPS=500
EVAL_SAMPLES=50

# Wandb configuration (set USE_WANDB=true to enable)
USE_WANDB=false
WANDB_PROJECT="biomamba2-training"
WANDB_ENTITY=""  # Leave empty for default
WANDB_RUN_NAME=""  # Leave empty for auto-generated

# Build wandb arguments
WANDB_ARGS=""
if [ "$USE_WANDB" = true ]; then
    WANDB_ARGS="--use_wandb"
    [ -n "$WANDB_PROJECT" ] && WANDB_ARGS="$WANDB_ARGS --wandb_project $WANDB_PROJECT"
    [ -n "$WANDB_ENTITY" ] && WANDB_ARGS="$WANDB_ARGS --wandb_entity $WANDB_ENTITY"
    [ -n "$WANDB_RUN_NAME" ] && WANDB_ARGS="$WANDB_ARGS --wandb_run_name $WANDB_RUN_NAME"
    echo "Wandb logging enabled (project: $WANDB_PROJECT)"
fi

# Run training
python scripts/training/finetune_pubmed_medline.py \
    --model_name $MODEL_NAME \
    --gpu_id $GPU_ID \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --accumulation_steps $ACCUMULATION_STEPS \
    --lr $LEARNING_RATE \
    --max_length $MAX_LENGTH \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --log_dir $LOG_DIR \
    --warmup_ratio $WARMUP_RATIO \
    --weight_decay $WEIGHT_DECAY \
    --max_grad_norm $MAX_GRAD_NORM \
    --scheduler $SCHEDULER \
    --logging_steps $LOGGING_STEPS \
    --eval_steps $EVAL_STEPS \
    --save_steps $SAVE_STEPS \
    --eval_samples $EVAL_SAMPLES \
    --use_title \
    --fp16 \
    --seed 42 \
    $WANDB_ARGS

echo "Training completed!"
