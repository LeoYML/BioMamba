#!/bin/bash

################################################################################
# Batch Evaluation Script - Compare Multiple Models on BioASQ
################################################################################

# Configuration
cd "$(dirname "$0")/../.."

DATASET="pubmedqa"  # or "bioasq"
SPLIT="validation"
MAX_SAMPLES=100  # Set to null for all samples

# Generation parameters
MAX_NEW_TOKENS=10
TEMPERATURE=0.1
TOP_P=0.9

# Hardware
GPU_ID=0

# Output
OUTPUT_DIR="./evaluation_results/comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

################################################################################
# Model configurations
################################################################################

# Add your models here (format: "model_type|model_path")
MODELS=(
    # Mamba2 models
    "mamba2|./checkpoints/biomamba2_sft_mamba2-130m_full_20260203_092105/best_model"
    
    # BioGPT models
    "biogpt|microsoft/biogpt"
    "biogpt|microsoft/BioGPT-Large"
    
    # GPT2 bio models
    "gpt2|stanford-crfm/BioMedLM"
    
    # Add more models as needed
)

################################################################################
# Run evaluation on all models
################################################################################

echo "=========================================="
echo "Multi-Model BioASQ Evaluation"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Split: $SPLIT"
echo "Number of models: ${#MODELS[@]}"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="
echo ""

RESULTS_FILE="$OUTPUT_DIR/summary.txt"
echo "Multi-Model Evaluation Results" > "$RESULTS_FILE"
echo "Dataset: $DATASET" >> "$RESULTS_FILE"
echo "Split: $SPLIT" >> "$RESULTS_FILE"
echo "Date: $(date)" >> "$RESULTS_FILE"
echo "==========================================" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

for MODEL_CONFIG in "${MODELS[@]}"; do
    IFS='|' read -r MODEL_TYPE MODEL_PATH <<< "$MODEL_CONFIG"
    
    echo ""
    echo "=========================================="
    echo "Evaluating: $MODEL_PATH"
    echo "Type: $MODEL_TYPE"
    echo "=========================================="
    echo ""
    
    # Create model-specific output directory
    MODEL_NAME=$(basename "$MODEL_PATH")
    MODEL_OUTPUT_DIR="$OUTPUT_DIR/${MODEL_TYPE}_${MODEL_NAME}"
    mkdir -p "$MODEL_OUTPUT_DIR"
    
    # Run evaluation
    python scripts/evaluation/evaluate_bioasq.py \
        --model_type "$MODEL_TYPE" \
        --model_path "$MODEL_PATH" \
        --dataset_name "$DATASET" \
        --split "$SPLIT" \
        --max_samples $MAX_SAMPLES \
        --max_new_tokens $MAX_NEW_TOKENS \
        --temperature $TEMPERATURE \
        --top_p $TOP_P \
        --gpu_id $GPU_ID \
        --output_dir "$MODEL_OUTPUT_DIR" \
        --save_predictions \
        --do_sample
    
    # Extract and append results to summary
    echo "Model: $MODEL_PATH ($MODEL_TYPE)" >> "$RESULTS_FILE"
    
    # Find the most recent metrics file
    METRICS_FILE=$(ls -t "$MODEL_OUTPUT_DIR"/*_metrics.json 2>/dev/null | head -n 1)
    
    if [ -f "$METRICS_FILE" ]; then
        echo "Metrics file: $METRICS_FILE" >> "$RESULTS_FILE"
        
        # Extract key metrics using Python
        python3 -c "
import json
with open('$METRICS_FILE', 'r') as f:
    data = json.load(f)
    metrics = data.get('metrics', {})
    print(f\"  Accuracy: {metrics.get('accuracy', 0):.4f}\")
    print(f\"  F1 Score: {metrics.get('f1', 0):.4f}\")
    print(f\"  Precision: {metrics.get('precision', 0):.4f}\")
    print(f\"  Recall: {metrics.get('recall', 0):.4f}\")
" >> "$RESULTS_FILE"
    else
        echo "  Error: Could not find metrics file" >> "$RESULTS_FILE"
    fi
    
    echo "" >> "$RESULTS_FILE"
    echo "------------------------------------------" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"
done

echo ""
echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="
echo "Summary saved to: $RESULTS_FILE"
echo ""
echo "Results:"
cat "$RESULTS_FILE"
