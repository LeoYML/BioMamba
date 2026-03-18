#!/bin/bash
# CPT for Mamba2-1.3b using mixed data (PubMed + 10% C4 + 10% Wikipedia)
# BS=24 peaks at ~65.2GB on H100 80GB

cd "$(dirname "$0")/../.."

source ./.venv/bin/activate

python scripts/training/run_cpt_singledoc.py \
    --model mamba2-1.3b \
    --data_dir ./data/pubmed_mixed_10pct_general_10pct_wiki \
    --epochs 3 \
    --lr 5e-6 \
    --batch_size 16 \
    --accum 16 \
    --label_smoothing 0.0 \
    --layer_lr_decay 0.95 \
    --output_dir ./checkpoints/1.3b_mixed_wiki \
    --gpu_id 0 \
    --seed 42
