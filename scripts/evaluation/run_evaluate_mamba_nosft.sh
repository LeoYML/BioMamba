#!/bin/bash

################################################################################
# BioASQ Evaluation Script for Mamba2 Models - Baseline Testing
# This script tests different Mamba2 model checkpoints
################################################################################

cd "$(dirname "$0")/../.."

set -e  # Exit on error

################################################################################
# Configuration
################################################################################

MODEL_TYPE="mamba2"
# Select model variant via env var to avoid accidental path overwrite.
# Usage examples:
#   MODEL_VARIANT=pretrain_best bash run_evaluate_mamba_nosft.sh
#   MODEL_VARIANT=sft_best bash run_evaluate_mamba_nosft.sh
MODEL_VARIANT="pretrain_ckpt1600_old"
case "$MODEL_VARIANT" in
  hf_mamba2)
    MODEL_PATH="state-spaces/mamba2-130m"
    MODEL_NAME="mamba2_original_hf"
    ;;
  hf_mamba1)
    MODEL_PATH="state-spaces/mamba-130m"
    MODEL_NAME="mamba1_original_hf"
    ;;
  pretrain_final_old)
    MODEL_PATH="./checkpoints/biomamba2_mamba2-130m_20260203_083540/final_model"
    MODEL_NAME="mamba2_bio_pretrain_final_old"
    ;;
  pretrain_best)
    MODEL_PATH="./checkpoints/biomamba2_mamba2-130m_20260209_002521/best_model"
    MODEL_NAME="mamba2_bio_pretrain_best"
    ;;
  pretrain_ckpt1600_old)
    MODEL_PATH="./checkpoints/biomamba2_mamba2-130m_20260209_025427/best_model"
    MODEL_NAME="mamba2_bio_pretrain_ckpt1600_old"
    ;;
  sft_best)
    MODEL_PATH="./checkpoints/biomamba2_sft_mamba2-130m_full_20260209_030622/best_model"
    MODEL_NAME="mamba2_sft_finetuned_best"
    ;;
  *)
    echo "ERROR: Unknown MODEL_VARIANT=$MODEL_VARIANT"
    echo "Valid values: hf_mamba2, hf_mamba1, pretrain_final_old, pretrain_best, pretrain_ckpt1600_old, sft_best"
    exit 1
    ;;
esac

# Optional explicit override for custom checkpoints.
if [[ -n "${MODEL_PATH:-}" ]]; then
  MODEL_NAME="${MODEL_NAME:-custom_model}"
  echo "Using explicit MODEL_PATH override: $MODEL_PATH"
fi

# Dataset configuration
DATASET="${DATASET:-bioasq}"  # "bioasq" or "pubmedqa"
DATA_PATH="${DATA_PATH:-./data/bioasq_test}"  # Local BioASQ test dataset
SPLIT="${SPLIT:-test}"
MAX_SAMPLES="${MAX_SAMPLES:-}"  # Empty for all samples (82), or set a number like "20" for quick test

# Generation parameters (optimized for yes/no questions)
MAX_LENGTH="${MAX_LENGTH:-512}"          # Maximum input sequence length
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-5}"    # Only need a few tokens for yes/no/maybe
TEMPERATURE="${TEMPERATURE:-0.0}"        # Ignored when DO_SAMPLE=false
TOP_P="${TOP_P:-0.9}"                    # Used only if DO_SAMPLE=true
DO_SAMPLE="${DO_SAMPLE:-false}"          # Greedy decoding for stable A/B comparison

# Hardware
GPU_ID="${GPU_ID:-0}"

# Output
OUTPUT_DIR="${OUTPUT_DIR:-./evaluation_results/mamba2_${MODEL_VARIANT}}"

################################################################################
# Validate paths
################################################################################

# Check if it's a HuggingFace model or local path
if [[ "$MODEL_PATH" == *"/"* ]] && [[ ! "$MODEL_PATH" == "./"* ]] && [[ ! -d "$MODEL_PATH" ]]; then
    # Looks like a HuggingFace model (e.g., state-spaces/mamba2-130m)
    echo "Note: Will download model from HuggingFace: $MODEL_PATH"
    echo ""
elif [[ "$MODEL_PATH" == "./"* ]] || [[ -d "$MODEL_PATH" ]]; then
    # Local path
    if [ ! -d "$MODEL_PATH" ]; then
        echo "=========================================="
        echo "ERROR: Model path not found!"
        echo "=========================================="
        echo "Path: $MODEL_PATH"
        echo ""
        echo "Available checkpoints:"
        ls -d ./checkpoints/*/ 2>/dev/null || echo "No checkpoints found"
        echo ""
        exit 1
    fi
else
    echo "Warning: Could not determine if path is local or HuggingFace: $MODEL_PATH"
fi

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
echo "Mamba Model Evaluation (No SFT)"
echo "=========================================="
echo "Model Name:    $MODEL_NAME"
echo "Model Type:    $MODEL_TYPE"
echo "Model Path:    $MODEL_PATH"
echo ""
if [[ "$MODEL_PATH" == "state-spaces/"* ]]; then
    echo "  Testing original HuggingFace model (no biomedical pre-training)"
    echo "  Note: Will use GPT-NeoX tokenizer (state-spaces models don't include tokenizer)"
elif [[ "$MODEL_PATH" == *"biomamba2_mamba2"* ]]; then
    echo "✓ Testing bio pre-trained model (no SFT)"
elif [[ "$MODEL_PATH" == *"sft"* ]]; then
    echo "✓ Testing SFT fine-tuned model"
fi
echo ""
echo "----------------------------------------"
echo "Dataset:       $DATASET"
echo "Data Path:     $DATA_PATH"
echo "Split:         $SPLIT"
echo "Max Samples:   ${MAX_SAMPLES:-All (82)}"
echo "Max Length:    $MAX_LENGTH"
echo "Max New Tokens: $MAX_NEW_TOKENS"
echo "Temperature:   $TEMPERATURE"
echo "Top-P:         $TOP_P"
echo "GPU ID:        $GPU_ID"
echo "Output Dir:    $OUTPUT_DIR"
echo "=========================================="
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
# Run evaluation
################################################################################

echo "Starting evaluation..."
echo ""

# Build command with optional parameters
CMD="python scripts/evaluation/evaluate_bioasq.py \
    --model_type $MODEL_TYPE \
    --model_path $MODEL_PATH \
    --dataset_name $DATASET \
    --data_path $DATA_PATH \
    --split $SPLIT \
    --max_length $MAX_LENGTH \
    --max_new_tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --gpu_id $GPU_ID \
    --output_dir $OUTPUT_DIR \
    --save_predictions"

# Add optional parameters
if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
fi

if [ "$DO_SAMPLE" = true ]; then
    CMD="$CMD --do_sample"
fi

# Execute
eval $CMD

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
    
    # Extract key metrics using Python
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
