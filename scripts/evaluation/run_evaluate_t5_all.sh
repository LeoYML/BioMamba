#!/bin/bash

################################################################################
# Evaluate Multiple T5 Medical Models
# Tests both FLAN-T5 Large and Small for medical QA
################################################################################

cd "$(dirname "$0")/../.."

set -e  # Exit on error

# Dataset configuration
DATASET="bioasq"
DATA_PATH="./data/bioasq_test"
SPLIT="test"
MAX_SAMPLES=""  # Empty for all (82), or set number for quick test

# Generation parameters
MAX_LENGTH=512
MAX_NEW_TOKENS=5
TEMPERATURE=0.7
TOP_P=0.9
GPU_ID=0

################################################################################
# Check dataset
################################################################################

if [ ! -d "$DATA_PATH" ]; then
    echo "ERROR: Dataset not found at $DATA_PATH"
    echo "Please run: bash setup_bioasq_test.sh"
    exit 1
fi

################################################################################
# Models to test
################################################################################

declare -A MODELS=(
    ["flan-t5-small"]="google/flan-t5-small"
    ["flan-t5-base"]="google/flan-t5-base"
    ["flan-t5-large"]="google/flan-t5-large"
    ["medical-qa-t5-lora"]="Adilbai/medical-qa-t5-lora"
)

# Order of evaluation (general models first, then specialized)
MODEL_ORDER=("flan-t5-small" "flan-t5-base" "flan-t5-large" "medical-qa-t5-lora")

################################################################################
# Evaluate each model
################################################################################

echo ""
echo "========================================================================"
echo "T5 Medical Models Evaluation on BioASQ"
echo "========================================================================"
echo ""
echo "Will evaluate ${#MODELS[@]} models:"
for model_name in "${MODEL_ORDER[@]}"; do
    echo "  - $model_name: ${MODELS[$model_name]}"
done
echo ""
echo "Dataset: $DATASET ($DATA_PATH)"
echo "Samples: ${MAX_SAMPLES:-All (82)}"
echo "========================================================================"
echo ""


for model_name in "${MODEL_ORDER[@]}"; do
    model_path="${MODELS[$model_name]}"
    output_dir="./evaluation_results/t5_${model_name}"
    
    echo ""
    echo "========================================================================"
    echo "Evaluating: $model_name"
    echo "========================================================================"
    echo "Model Path: $model_path"
    echo "Output Dir: $output_dir"
    echo "------------------------------------------------------------------------"
    echo ""
    
    # Run evaluation
    python scripts/evaluation/evaluate_bioasq.py \
        --model_type t5 \
        --model_path "$model_path" \
        --dataset_name "$DATASET" \
        --data_path "$DATA_PATH" \
        --split "$SPLIT" \
        ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES} \
        --max_length $MAX_LENGTH \
        --max_new_tokens $MAX_NEW_TOKENS \
        --temperature $TEMPERATURE \
        --top_p $TOP_P \
        --gpu_id $GPU_ID \
        --output_dir "$output_dir" \
        --save_predictions \
        --do_sample
    
    echo ""
    echo "✓ Completed: $model_name"
    echo ""
done

################################################################################
# Summary
################################################################################

echo ""
echo "========================================================================"
echo "EVALUATION SUMMARY"
echo "========================================================================"
echo ""

for model_name in "${MODEL_ORDER[@]}"; do
    output_dir="./evaluation_results/t5_${model_name}"
    latest_metrics=$(ls -t "$output_dir"/*_metrics.json 2>/dev/null | head -n 1)
    
    if [ -f "$latest_metrics" ]; then
        echo "Model: $model_name"
        echo "----------------------------------------------------------------------"
        python3 << EOF
import json
try:
    with open("$latest_metrics", "r") as f:
        data = json.load(f)
    metrics = data.get("metrics", {})
    print(f"  Accuracy:      {metrics.get('accuracy', 0):.4f} ({metrics.get('accuracy', 0)*100:.2f}%)")
    print(f"  F1 Score:      {metrics.get('f1', 0):.4f}")
    print(f"  Valid Samples: {metrics.get('valid_samples', 0)}/{metrics.get('total_samples', 0)}")
except:
    print("  Could not load metrics")
EOF
        echo ""
    else
        echo "Model: $model_name"
        echo "  No results found"
        echo ""
    fi
done

echo "========================================================================"
echo "All evaluations completed!"
echo "========================================================================"
