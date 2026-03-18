#!/bin/bash
# Sequential CPT training: wait for 780m → run 1.3b → run 370m
cd "$(dirname "$0")/../.."

source ./.venv/bin/activate

echo "$(date): Waiting for 780m training (PID 79387) to finish..."
while kill -0 79387 2>/dev/null; do
    sleep 60
done
echo "$(date): 780m training finished."

echo "$(date): Starting 1.3b CPT..."
python scripts/training/run_cpt_singledoc.py \
    --model mamba2-1.3b \
    --data_dir ./data/pubmed_mixed_10pct_general_10pct_wiki \
    --epochs 3 \
    --lr 5e-6 \
    --batch_size 24 \
    --accum 10 \
    --label_smoothing 0.0 \
    --layer_lr_decay 0.9 \
    --output_dir ./checkpoints/1.3b_mixed_wiki \
    --gpu_id 0 \
    --seed 42
echo "$(date): 1.3b training finished."

echo "$(date): Starting 370m CPT..."
python scripts/training/run_cpt_singledoc.py \
    --model mamba2-370m \
    --data_dir ./data/pubmed_mixed_10pct_general_10pct_wiki \
    --epochs 3 \
    --lr 5e-6 \
    --batch_size 48 \
    --accum 5 \
    --label_smoothing 0.0 \
    --layer_lr_decay 0.9 \
    --output_dir ./checkpoints/370m_mixed_wiki \
    --gpu_id 0 \
    --seed 42
echo "$(date): 370m training finished. All done!"
