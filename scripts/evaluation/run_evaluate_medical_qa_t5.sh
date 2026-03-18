#!/bin/bash

################################################################################
# BioASQ Evaluation Script for Medical-QA-T5 Models
# Tests both large and small versions of Medical-QA-T5
################################################################################

cd "$(dirname "$0")/../.."

set -e  # Exit on error

################################################################################
# Configuration
################################################################################

# Model Selection - Choose which Medical-QA-T5 model to evaluate

# Option 1: Medical-QA-T5 Large (RECOMMENDED)
MODEL_PATH="google/flan-t5-large"  # Base T5, will be fine-tuned for medical QA
MODEL_NAME="medical-qa-t5-large"
MODEL_SIZE="large"

# Option 2: Medical-QA-T5 Small (Faster)
# MODEL_PATH="google/flan-t5-small"
# MODEL_NAME="medical-qa-t5-small"
# MODEL_SIZE="small"

# Option 3: Medical-QA-T5 Base (Medium)
# MODEL_PATH="google/flan-t5-base"
# MODEL_NAME="medical-qa-t5-base"
# MODEL_SIZE="base"

# Actual Medical-QA models (if they exist on HuggingFace)
# MODEL_PATH="medicalai/medical-qa-t5-large"
# MODEL_PATH="medicalai/medical-qa-t5-small"

MODEL_TYPE="t5"  # Use T5 model type

# Dataset configuration
DATASET="bioasq"
DATA_PATH="./data/bioasq_test"
SPLIT="test"
MAX_SAMPLES=""  # Empty for all samples (82), or set a number for quick test

# Generation parameters for T5
MAX_LENGTH=512
MAX_NEW_TOKENS=5      # Only need a few tokens for yes/no/maybe
TEMPERATURE=0.7
TOP_P=0.9

# Hardware
GPU_ID=0

# Output - separate directories for different model sizes
OUTPUT_DIR="./evaluation_results/medical_qa_t5_${MODEL_SIZE}"

################################################################################
# Validate dataset
################################################################################

if [ ! -d "$DATA_PATH" ]; then
    echo "=========================================="
    echo "ERROR: Dataset not found!"
    echo "=========================================="
    echo "Path: $DATA_PATH"
    echo ""
    echo "Please run setup first:"
    echo "  bash setup_bioasq_test.sh"
    echo ""
    exit 1
fi

################################################################################
# Display configuration
################################################################################

echo ""
echo "=========================================="
echo "Medical-QA-T5 Model Evaluation"
echo "=========================================="
echo "Model:        $MODEL_PATH"
echo "Model Name:   $MODEL_NAME"
echo "Model Size:   $MODEL_SIZE"
echo ""
echo "Dataset:      $DATASET"
echo "Data Path:    $DATA_PATH"
echo "Split:        $SPLIT"
echo "Max Samples:  ${MAX_SAMPLES:-All (82)}"
echo "Max Length:   $MAX_LENGTH"
echo "Output Dir:   $OUTPUT_DIR"
echo "=========================================="
echo ""

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | head -n 1
    echo ""
fi

################################################################################
# Run evaluation
################################################################################

echo "Starting evaluation..."
echo ""

python scripts/evaluation/evaluate_bioasq.py \
    --model_type "$MODEL_TYPE" \
    --model_path "$MODEL_PATH" \
    --dataset_name "$DATASET" \
    --data_path "$DATA_PATH" \
    --split "$SPLIT" \
    ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES} \
    --max_length $MAX_LENGTH \
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
