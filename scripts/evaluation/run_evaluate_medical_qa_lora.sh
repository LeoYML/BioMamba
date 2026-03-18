#!/bin/bash

################################################################################
# BioASQ Evaluation Script for Adilbai/medical-qa-t5-lora
# Medical QA T5 model with LoRA fine-tuning
################################################################################

cd "$(dirname "$0")/../.."

set -e  # Exit on error

################################################################################
# Configuration
################################################################################

MODEL_TYPE="t5"
MODEL_PATH="Adilbai/medical-qa-t5-lora"
MODEL_NAME="medical-qa-t5-lora"

# Dataset configuration
DATASET="bioasq"
DATA_PATH="./data/bioasq_test"
SPLIT="test"
MAX_SAMPLES=""  # Empty for all samples (82), or set a number for quick test

# Generation parameters for T5
MAX_LENGTH=512
MAX_NEW_TOKENS=10     # Increased to allow for complete answers
TEMPERATURE=0.3       # Lower temperature for more deterministic answers
TOP_P=0.9

# Hardware
GPU_ID=0

# Output
OUTPUT_DIR="./evaluation_results/medical_qa_t5_lora"

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
echo "Medical-QA-T5-LoRA Model Evaluation"
echo "=========================================="
echo "Model:        $MODEL_PATH"
echo "Model Name:   $MODEL_NAME"
echo "Type:         T5 with LoRA fine-tuning"
echo "Purpose:      Medical Question Answering"
echo ""
echo "Dataset:      $DATASET"
echo "Data Path:    $DATA_PATH"
echo "Split:        $SPLIT"
echo "Max Samples:  ${MAX_SAMPLES:-All (82)}"
echo "Max Length:   $MAX_LENGTH"
echo "Max New Tokens: $MAX_NEW_TOKENS"
echo "Temperature:  $TEMPERATURE"
echo "Output Dir:   $OUTPUT_DIR"
echo "=========================================="
echo ""

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | head -n 1
    echo ""
fi

echo "📥 Note: Model will be downloaded from HuggingFace on first run"
echo "   This may take a few minutes..."
echo ""

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

################################################################################
# Post-evaluation summary
################################################################################

echo ""
echo "=========================================="
echo "Evaluation Completed!"
echo "=========================================="
echo ""

# Find the latest results file
LATEST_METRICS=$(ls -t "$OUTPUT_DIR"/*_metrics.json 2>/dev/null | head -n 1)
LATEST_PREDICTIONS=$(ls -t "$OUTPUT_DIR"/*_predictions.json 2>/dev/null | head -n 1)

if [ -f "$LATEST_METRICS" ]; then
    echo "Results Summary:"
    echo "----------------"
    echo "Metrics file: $LATEST_METRICS"
    echo "Predictions file: $LATEST_PREDICTIONS"
    echo ""
    
    # Extract key metrics
    python3 << EOF
import json
import sys

try:
    with open("$LATEST_METRICS", "r") as f:
        data = json.load(f)
    
    metrics = data.get("metrics", {})
    print(f"Accuracy:      {metrics.get('accuracy', 0):.4f} ({metrics.get('accuracy', 0)*100:.2f}%)")
    print(f"F1 Score:      {metrics.get('f1', 0):.4f}")
    print(f"Precision:     {metrics.get('precision', 0):.4f}")
    print(f"Recall:        {metrics.get('recall', 0):.4f}")
    print(f"Valid Samples: {metrics.get('valid_samples', 0)}/{metrics.get('total_samples', 0)}")
    print(f"Error Samples: {metrics.get('error_samples', 0)}")
    
    # Per-class metrics
    per_class = metrics.get('per_class', {})
    if per_class:
        print("\nPer-class Performance:")
        for label, class_metrics in per_class.items():
            print(f"  {label:8s} - F1: {class_metrics.get('f1', 0):.4f}, Support: {class_metrics.get('support', 0)}")
    
except Exception as e:
    print(f"Could not parse metrics: {e}")
    sys.exit(0)
EOF
    
    echo ""
    echo "Full results saved to: $OUTPUT_DIR"
else
    echo "No metrics file found. Check for errors above."
fi

echo ""
echo "=========================================="
