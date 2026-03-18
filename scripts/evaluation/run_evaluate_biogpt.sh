#!/bin/bash

################################################################################
# BioASQ Evaluation Script for BioGPT Model
################################################################################

# Model Selection - Choose which BioGPT model to evaluate

# Option 1: BioGPT-Large (1.5B parameters) - RECOMMENDED for best performance
cd "$(dirname "$0")/../.."

MODEL_PATH="microsoft/biogpt-large"
MODEL_NAME="biogpt-large-1.5b"

# Option 2: BioGPT-Base (350M parameters) - Faster, less memory
# MODEL_PATH="microsoft/biogpt"
# MODEL_NAME="biogpt-base-350m"

# Configuration
MODEL_TYPE="biogpt"

# Dataset configuration
DATASET="bioasq"  # "bioasq" or "pubmedqa"
DATA_PATH="./data/bioasq_test"  # Local BioASQ test dataset from golden files
SPLIT="test"  # "train", "validation", or "test"
MAX_SAMPLES=""  # Set to empty string "" for all samples (82 total)

# Generation parameters
MAX_LENGTH=512  # Reduced from 1024 to prevent CUDA errors with BioGPT
MAX_NEW_TOKENS=3   # Only need 1-2 tokens for yes/no (reduced from 10)
TEMPERATURE=0.7    # Increased from 0.1 for more diverse generation
TOP_P=0.9

# Hardware
GPU_ID=0

# Output - separate directories for different model sizes
if [[ "$MODEL_PATH" == *"large"* ]]; then
    OUTPUT_DIR="./evaluation_results/biogpt_large"
else
    OUTPUT_DIR="./evaluation_results/biogpt"
fi

################################################################################
# Check dataset exists
################################################################################

if [ ! -d "$DATA_PATH" ]; then
    echo "=========================================="
    echo "Error: Dataset not found"
    echo "=========================================="
    echo ""
    echo "Dataset path: $DATA_PATH"
    echo ""
    echo "Please run setup first:"
    echo "  bash setup_bioasq_test.sh"
    echo ""
    exit 1
fi

################################################################################
# Run evaluation
################################################################################

echo ""
echo "=========================================="
echo "BioASQ Evaluation - BioGPT Model"
echo "=========================================="
echo "Model:        $MODEL_PATH"
echo "Model Size:   $MODEL_NAME"
if [[ "$MODEL_PATH" == *"large"* ]]; then
    echo "Parameters:   ~1.5B"
    echo "Note:         Large model requires more GPU memory (~16GB)"
else
    echo "Parameters:   ~350M"
fi
echo ""
echo "Dataset:      $DATASET"
echo "Data path:    $DATA_PATH"
echo "Split:        $SPLIT"
echo "Max samples:  ${MAX_SAMPLES:-All (82)}"
echo "Max length:   $MAX_LENGTH"
echo "Output dir:   $OUTPUT_DIR"
echo "=========================================="
echo ""

# Check GPU memory if using large model
if [[ "$MODEL_PATH" == *"large"* ]] && command -v nvidia-smi &> /dev/null; then
    echo "GPU Memory Check:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | head -n 1
    echo ""
fi

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
