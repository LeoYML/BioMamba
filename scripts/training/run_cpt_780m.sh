#!/bin/bash
# CPT for Mamba2-780m using mixed data (PubMed + 10% C4 + 10% Wikipedia)
# BS=32 peaks at ~64.5GB on H100 80GB

cd "$(dirname "$0")/../.."

source ./.venv/bin/activate

python scripts/training/run_cpt_singledoc.py \
    --model mamba2-780m \
    --data_dir ./data/pubmed_mixed_10pct_general_10pct_wiki \
    --epochs 3 \
    --lr 5e-6 \
    --batch_size 32 \
    --accum 8 \
    --label_smoothing 0.0 \
    --layer_lr_decay 0.95 \
    --output_dir ./checkpoints/780m_mixed_wiki \
    --gpu_id 0 \
    --seed 42
