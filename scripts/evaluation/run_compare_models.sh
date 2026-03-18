#!/bin/bash

################################################################################
# Compare Mamba2 SFT Model vs BioGPT on BioASQ Test Dataset
################################################################################

cd "$(dirname "$0")/../.."

echo "=========================================="
echo "Model Comparison on BioASQ Test Dataset"
echo "=========================================="
echo ""
echo "This script will evaluate:"
echo "  1. Mamba2 SFT model"
echo "  2. BioGPT (microsoft/biogpt)"
echo ""
echo "Dataset: BioASQ Test (82 yes/no questions)"
echo "=========================================="
echo ""

# Check if dataset exists
if [ ! -d "./data/bioasq_test" ]; then
    echo "Error: BioASQ test dataset not found!"
    echo "Please run: bash setup_bioasq_test.sh"
    exit 1
fi

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./evaluation_results/comparison_$TIMESTAMP"
mkdir -p "$OUTPUT_DIR"

echo "Results will be saved to: $OUTPUT_DIR"
echo ""

################################################################################
# Evaluate Mamba2 SFT Model
################################################################################

echo ""
echo "========================================"
echo "Step 1/2: Evaluating Mamba2 SFT Model"
echo "========================================"
echo ""

python scripts/evaluation/evaluate_bioasq.py \
    --model_type mamba2 \
    --model_path ./checkpoints/biomamba2_sft_mamba2-130m_full_20260203_092105/final_model \
    --dataset_name bioasq \
    --data_path ./data/bioasq_test \
    --split test \
    --max_new_tokens 10 \
    --temperature 0.1 \
    --top_p 0.9 \
    --gpu_id 0 \
    --output_dir "$OUTPUT_DIR/mamba2" \
    --save_predictions \
    --do_sample

if [ $? -ne 0 ]; then
    echo "Error: Mamba2 evaluation failed!"
    exit 1
fi

echo ""
echo "✓ Mamba2 evaluation completed"
echo ""

################################################################################
# Evaluate BioGPT Model
################################################################################

echo ""
echo "========================================"
echo "Step 2/2: Evaluating BioGPT Model"
echo "========================================"
echo ""

python scripts/evaluation/evaluate_bioasq.py \
    --model_type biogpt \
    --model_path microsoft/biogpt \
    --dataset_name bioasq \
    --data_path ./data/bioasq_test \
    --split test \
    --max_new_tokens 10 \
    --temperature 0.1 \
    --top_p 0.9 \
    --gpu_id 0 \
    --output_dir "$OUTPUT_DIR/biogpt" \
    --save_predictions \
    --do_sample

if [ $? -ne 0 ]; then
    echo "Error: BioGPT evaluation failed!"
    exit 1
fi

echo ""
echo "✓ BioGPT evaluation completed"
echo ""

################################################################################
# Generate Comparison Report
################################################################################

echo ""
echo "========================================"
echo "Generating Comparison Report"
echo "========================================"
echo ""

python scripts/analysis/analyze_evaluation_results.py \
    --results_dir "$OUTPUT_DIR" \
    --output_file "$OUTPUT_DIR/comparison_report.txt"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Comparison report generated"
    echo ""
    echo "========================================"
    echo "Results Summary"
    echo "========================================"
    echo ""
    cat "$OUTPUT_DIR/comparison_report.txt"
else
    echo ""
    echo "Note: Could not generate comparison report"
    echo "Manual comparison:"
    echo ""
    echo "Mamba2 results:"
    find "$OUTPUT_DIR/mamba2" -name "*_metrics.json" -exec echo "  {}" \;
    echo ""
    echo "BioGPT results:"
    find "$OUTPUT_DIR/biogpt" -name "*_metrics.json" -exec echo "  {}" \;
fi

echo ""
echo "========================================"
echo "Comparison Complete!"
echo "========================================"
echo ""
echo "All results saved to: $OUTPUT_DIR"
echo ""
