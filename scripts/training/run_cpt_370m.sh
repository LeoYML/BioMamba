#!/bin/bash
# CPT for Mamba2-370m using mixed data (PubMed + 10% C4 + 10% Wikipedia)
# BS=48 peaks at ~76.4GB on H100 80GB

cd "$(dirname "$0")/../.."

source ./.venv/bin/activate

python scripts/training/run_cpt_singledoc.py \
    --model mamba2-370m \
    --data_dir ./data/pubmed_mixed_10pct_general_10pct_wiki \
    --epochs 3 \
    --lr 5e-6 \
    --batch_size 48 \
    --accum 5 \
    --label_smoothing 0.0 \
    --layer_lr_decay 0.95 \
    --output_dir ./checkpoints/370m_mixed_wiki \
    --gpu_id 0 \
    --seed 42
