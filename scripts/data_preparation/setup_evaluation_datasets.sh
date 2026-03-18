#!/bin/bash

################################################################################
# Setup Evaluation Datasets
# This script downloads and prepares datasets for BioASQ evaluation
################################################################################

cd "$(dirname "$0")/../.."

echo "=========================================="
echo "Setting Up Evaluation Datasets"
echo "=========================================="
echo ""

# Configuration
OUTPUT_DIR="./data/evaluation_datasets"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Download PubMedQA
echo "=========================================="
echo "Downloading PubMedQA Dataset"
echo "=========================================="
echo ""

python scripts/data_preparation/download_datasets.py --output_dir "$OUTPUT_DIR" --dataset pubmedqa

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ PubMedQA downloaded successfully"
    PUBMEDQA_PATH="$OUTPUT_DIR/pubmedqa_pqa_labeled"
    echo "  Path: $PUBMEDQA_PATH"
else
    echo "✗ Failed to download PubMedQA"
    exit 1
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Downloaded datasets:"
echo "  - PubMedQA: $PUBMEDQA_PATH"
echo ""
echo "To evaluate with downloaded dataset:"
echo ""
echo "  python scripts/evaluation/evaluate_bioasq.py \\"
echo "      --model_type mamba2 \\"
echo "      --model_path ./checkpoints/your_model \\"
echo "      --dataset_name pubmedqa \\"
echo "      --data_path $PUBMEDQA_PATH"
echo ""
echo "Or update run_evaluate_bioasq.sh:"
echo "  DATA_PATH=\"$PUBMEDQA_PATH\""
echo ""
