#!/bin/bash

################################################################################
# Quick BioASQ Evaluation Script (5 samples for testing)
################################################################################

# Configuration
cd "$(dirname "$0")/../.."

MODEL_TYPE="mamba2"
MODEL_PATH="./checkpoints/biomamba2_sft_mamba2-130m_full_20260203_092105/final_model"

# Dataset configuration
DATASET="bioasq"
DATA_PATH="./data/bioasq_test"
SPLIT="test"
MAX_SAMPLES=5  # Quick test with 5 samples

# Generation parameters
MAX_NEW_TOKENS=10
TEMPERATURE=0.1
TOP_P=0.9

# Hardware
GPU_ID=0

# Output
OUTPUT_DIR="./test_results/bioasq_quick"

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

echo "=========================================="
echo "Quick BioASQ Evaluation (5 samples)"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET"
echo "Data path: $DATA_PATH"
echo "Split: $SPLIT"
echo "Samples: $MAX_SAMPLES"
echo "=========================================="
echo ""

python scripts/evaluation/evaluate_bioasq.py \
    --model_type "$MODEL_TYPE" \
    --model_path "$MODEL_PATH" \
    --dataset_name "$DATASET" \
    --data_path "$DATA_PATH" \
    --split "$SPLIT" \
    --max_samples $MAX_SAMPLES \
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
echo ""
echo "To run full evaluation:"
echo "  bash run_evaluate_bioasq.sh"
echo ""
