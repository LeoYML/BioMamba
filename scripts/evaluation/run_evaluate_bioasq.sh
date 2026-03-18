#!/bin/bash

################################################################################
# BioASQ Evaluation Script for Mamba2 SFT Model
################################################################################

# Configuration
cd "$(dirname "$0")/../.."

MODEL_TYPE="mamba2"
# Use the SFT model from the latest training run
MODEL_PATH="./checkpoints/biomamba2_sft_mamba2-130m_full_20260209_030622/best_model"

# Dataset configuration
DATASET="bioasq"  # "bioasq" or "pubmedqa"
DATA_PATH="./data/bioasq_test"  # Local BioASQ test dataset from golden files
SPLIT="test"  # "train", "validation", or "test"
MAX_SAMPLES=100 # Set to empty string "" for all samples

# Generation parameters
MAX_NEW_TOKENS=10
TEMPERATURE=0.1
TOP_P=0.9

# Hardware
GPU_ID=0

# Output
OUTPUT_DIR="./evaluation_results/mamba2"

################################################################################
# Run evaluation
################################################################################

echo "=========================================="
echo "BioASQ Evaluation - Mamba2 Model"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET"
echo "Data path: $DATA_PATH"
echo "Split: $SPLIT"
echo "Max samples: ${MAX_SAMPLES:-All}"
echo "=========================================="
echo ""

python scripts/evaluation/evaluate_bioasq.py \
    --model_type "$MODEL_TYPE" \
    --model_path "$MODEL_PATH" \
    --dataset_name "$DATASET" \
    --split "$SPLIT" \
    ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES} \
    ${DATA_PATH:+--data_path "$DATA_PATH"} \
    --max_new_tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --gpu_id $GPU_ID \
    --output_dir "$OUTPUT_DIR" \
    --save_predictions \
    --do_sample

echo ""
echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"
