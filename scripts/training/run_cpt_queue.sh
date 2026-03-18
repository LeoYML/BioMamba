#!/bin/bash
# Sequential CPT: 1.3b → 370m
cd "$(dirname "$0")/../.."

source ./.venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "$(date): Starting 1.3b CPT..."
python scripts/training/run_cpt_singledoc.py \
    --model mamba2-1.3b \
    --data_dir ./data/pubmed_mixed_10pct_general_10pct_wiki \
    --epochs 3 \
    --lr 5e-6 \
    --batch_size 24 \
    --accum 10 \
    --label_smoothing 0.0 \
    --layer_lr_decay 0.95 \
    --output_dir ./checkpoints/1.3b_mixed_wiki \
    --gpu_id 0 \
    --seed 42
echo "$(date): 1.3b training finished (exit code: $?)."

sleep 10

echo "$(date): Starting 370m CPT..."
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
echo "$(date): 370m training finished (exit code: $?). All done!"
