#!/bin/bash

################################################################################
# Quick Test - T5 Models (Small and Large only)
# Tests with 10 samples for quick verification
################################################################################

cd "$(dirname "$0")/../.."

set -e

echo ""
echo "=========================================="
echo "Quick T5 Models Test (10 samples)"
echo "=========================================="
echo ""

# Test FLAN-T5 Small
echo "Testing FLAN-T5 Small..."
python scripts/evaluation/evaluate_bioasq.py \
    --model_type t5 \
    --model_path "google/flan-t5-small" \
    --dataset_name bioasq \
    --data_path ./data/bioasq_test \
    --split test \
    --max_samples 10 \
    --max_length 512 \
    --max_new_tokens 5 \
    --temperature 0.7 \
    --top_p 0.9 \
    --gpu_id 0 \
    --output_dir ./evaluation_results/t5_flan-t5-small \
    --save_predictions \
    --do_sample

echo ""
echo "✓ FLAN-T5 Small completed"
echo ""

# Test FLAN-T5 Large
echo "Testing FLAN-T5 Large..."
python scripts/evaluation/evaluate_bioasq.py \
    --model_type t5 \
    --model_path "google/flan-t5-large" \
    --dataset_name bioasq \
    --data_path ./data/bioasq_test \
    --split test \
    --max_samples 10 \
    --max_length 512 \
    --max_new_tokens 5 \
    --temperature 0.7 \
    --top_p 0.9 \
    --gpu_id 0 \
    --output_dir ./evaluation_results/t5_flan-t5-large \
    --save_predictions \
    --do_sample

echo ""
echo "✓ FLAN-T5 Large completed"
echo ""
echo "=========================================="
echo "Quick test completed!"
echo "=========================================="
