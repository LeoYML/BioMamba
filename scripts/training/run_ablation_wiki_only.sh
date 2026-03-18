#!/bin/bash
# Ablation: 130m CPT with PubMed + Wiki only (no C4) at 10%, 20%, 30%
# Compare PubMed PPL against current best (10%C4+10%Wiki, val_loss=2.111)

cd "$(dirname "$0")/../.."

source ./.venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================="
echo "Step 1: Prepare wiki-only datasets"
echo "=========================================="
python scripts/data_preparation/prepare_wiki_only_data.py

echo ""
echo "=========================================="
echo "Step 2: CPT 130m with 10% Wiki (no C4)"
echo "=========================================="
python scripts/training/run_cpt_singledoc.py \
    --model mamba2-130m \
    --data_dir ./data/pubmed_wiki_only_10pct \
    --epochs 3 \
    --lr 5e-6 \
    --batch_size 32 \
    --accum 8 \
    --label_smoothing 0.0 \
    --layer_lr_decay 0.9 \
    --output_dir ./checkpoints/ablation_wiki10 \
    --gpu_id 0 \
    --seed 42

echo ""
echo "=========================================="
echo "Step 3: CPT 130m with 20% Wiki (no C4)"
echo "=========================================="
python scripts/training/run_cpt_singledoc.py \
    --model mamba2-130m \
    --data_dir ./data/pubmed_wiki_only_20pct \
    --epochs 3 \
    --lr 5e-6 \
    --batch_size 32 \
    --accum 8 \
    --label_smoothing 0.0 \
    --layer_lr_decay 0.9 \
    --output_dir ./checkpoints/ablation_wiki20 \
    --gpu_id 0 \
    --seed 42

echo ""
echo "=========================================="
echo "Step 4: CPT 130m with 30% Wiki (no C4)"
echo "=========================================="
python scripts/training/run_cpt_singledoc.py \
    --model mamba2-130m \
    --data_dir ./data/pubmed_wiki_only_30pct \
    --epochs 3 \
    --lr 5e-6 \
    --batch_size 32 \
    --accum 8 \
    --label_smoothing 0.0 \
    --layer_lr_decay 0.9 \
    --output_dir ./checkpoints/ablation_wiki30 \
    --gpu_id 0 \
    --seed 42

echo ""
echo "=========================================="
echo "Step 5: Evaluate PubMed PPL for all models"
echo "=========================================="
python scripts/evaluation/eval_ablation_wiki.py

echo ""
echo "All ablation experiments complete!"
