#!/bin/bash

################################################################################
# Quick Test Script for BioASQ Evaluation
# This script runs a fast test with a small number of samples
################################################################################

cd "$(dirname "$0")/../.."

echo "=========================================="
echo "Quick Evaluation Test"
echo "=========================================="
echo ""

# Check if model exists
MODEL_PATH="./checkpoints/biomamba2_sft_mamba2-130m_full_20260203_092105/final_model"

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo ""
    echo "Available models:"
    ls -d ./checkpoints/*/final_model 2>/dev/null || echo "  No models found"
    echo ""
    echo "Please update MODEL_PATH in this script"
    exit 1
fi

echo "Model: $MODEL_PATH"
echo ""

# Test 1: Check environment
echo "Test 1: Checking environment..."
python tests/test_evaluation.py --model_path "$MODEL_PATH" --model_type mamba2 --skip_generation

if [ $? -ne 0 ]; then
    echo "✗ Environment test failed"
    exit 1
fi

echo ""
echo "✓ Environment test passed"
echo ""

# Test 2: Quick evaluation with 5 samples
echo "Test 2: Running quick evaluation (5 samples)..."
python scripts/evaluation/evaluate_bioasq.py \
    --model_type mamba2 \
    --model_path "$MODEL_PATH" \
    --dataset_name pubmedqa \
    --split validation \
    --max_samples 5 \
    --max_new_tokens 10 \
    --temperature 0.1 \
    --gpu_id 0 \
    --output_dir ./test_results \
    --save_predictions \
    --do_sample

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Quick evaluation passed"
    echo ""
    echo "Results saved to: ./test_results/"
    echo ""
    echo "To run full evaluation, use:"
    echo "  bash run_evaluate_bioasq.sh"
else
    echo "✗ Quick evaluation failed"
    exit 1
fi
