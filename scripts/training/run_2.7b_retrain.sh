#!/bin/bash
# Retrain 2.7b base+SFT with lower lr to fix overfitting
# Original: lr=1e-5, epochs=3, batch=1, accum=32 → discharge ROUGE drops vs 1.3b
# Fix: lr=5e-6, epochs=2, batch=8, accum=4 (fill GPU)
cd "$(dirname "$0")/../.."

set -e

echo "$(date) Starting 2.7b base+SFT retrain (lr=5e-6, epochs=2)..."
python -m ft_biomamba.run_mimic_sft \
  --model mamba2-2.7b \
  --model_path state-spaces/mamba2-2.7b \
  --task both --epochs 2 --lr 5e-6 \
  --max_train_samples 20000 \
  --batch_size 4 --accum 8 \
  --warmup_ratio 0.1 \
  --output_dir ./checkpoints/mimic_sft_2.7b_v2

echo "$(date) 2.7b retrain complete!"
