#!/bin/bash

################################################################################
# Setup BioASQ Test Dataset from Golden Files
# Converts 13B1-13B4 golden JSON files to HuggingFace format
################################################################################

cd "$(dirname "$0")/../.."

echo "=========================================="
echo "Setting Up BioASQ Test Dataset"
echo "=========================================="
echo ""

# Configuration
GOLDEN_FILES=(
    "13B1_golden.json"
    "13B2_golden.json"
    "13B3_golden.json"
    "13B4_golden.json"
)

OUTPUT_DIR="./data/bioasq_test"
QUESTION_TYPE="yesno"  # Only yes/no questions for evaluation

# Check if files exist
echo "Checking for golden files..."
MISSING_FILES=()
for file in "${GOLDEN_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        MISSING_FILES+=("$file")
    else
        echo "  ✓ Found: $file"
    fi
done

if [ ${#MISSING_FILES[@]} -ne 0 ]; then
    echo ""
    echo "Error: Missing files:"
    for file in "${MISSING_FILES[@]}"; do
        echo "  ✗ $file"
    done
    echo ""
    echo "Please ensure all BioASQ golden files are in the current directory."
    exit 1
fi

echo ""
echo "All golden files found!"
echo ""

# Convert to HuggingFace format
echo "=========================================="
echo "Converting to HuggingFace Dataset Format"
echo "=========================================="
echo ""

python scripts/data_preparation/convert_bioasq_golden_to_dataset.py \
    --input_files "${GOLDEN_FILES[@]}" \
    --output_dir "$OUTPUT_DIR" \
    --question_type "$QUESTION_TYPE"

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ Conversion failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "BioASQ test dataset ready at: $OUTPUT_DIR"
echo ""
echo "Quick test (5 samples):"
echo "  python scripts/evaluation/evaluate_bioasq.py \\"
echo "      --model_type mamba2 \\"
echo "      --model_path ./checkpoints/biomamba2_sft_mamba2-130m_full_20260203_092105/final_model \\"
echo "      --dataset_name bioasq \\"
echo "      --data_path $OUTPUT_DIR \\"
echo "      --split test \\"
echo "      --max_samples 5 \\"
echo "      --save_predictions"
echo ""
echo "Full evaluation:"
echo "  python scripts/evaluation/evaluate_bioasq.py \\"
echo "      --model_type mamba2 \\"
echo "      --model_path ./checkpoints/biomamba2_sft_mamba2-130m_full_20260203_092105/final_model \\"
echo "      --dataset_name bioasq \\"
echo "      --data_path $OUTPUT_DIR \\"
echo "      --split test \\"
echo "      --save_predictions"
echo ""
