#!/bin/bash
# Train CPT+SFT for all sizes (370m, 780m, 1.3b, 2.7b)
# Uses CPT (PubMed+Wiki) checkpoints as starting point for clinical SFT
# Batch sizes optimized for H100 80GB (tested with optimizer states)
cd "$(dirname "$0")/../.."

set -e

echo "$(date) === CPT+SFT Training for all sizes ==="

# 370m CPT+SFT — max batch=48, use 32 for safety, accum=1
echo "$(date) Starting 370m CPT+SFT..."
python -m ft_biomamba.run_mimic_sft \
  --model mamba2-370m \
  --model_path ./checkpoints/370m_mixed_wiki/biomamba_cpt_singledoc_mamba2-370m/best_model \
  --task both --epochs 3 --lr 2e-5 \
  --max_train_samples 20000 \
  --batch_size 32 --accum 1 \
  --output_dir ./checkpoints/mimic_sft_370m_cpt

# 780m CPT+SFT — max batch=32, use 16 for safety, accum=2
echo "$(date) Starting 780m CPT+SFT..."
python -m ft_biomamba.run_mimic_sft \
  --model mamba2-780m \
  --model_path ./checkpoints/780m_mixed_wiki/biomamba_cpt_singledoc_mamba2-780m/best_model \
  --task both --epochs 3 --lr 2e-5 \
  --max_train_samples 20000 \
  --batch_size 16 --accum 2 \
  --output_dir ./checkpoints/mimic_sft_780m_cpt

# 1.3b CPT+SFT — max batch=24, use 16 for safety, accum=2
echo "$(date) Starting 1.3b CPT+SFT..."
python -m ft_biomamba.run_mimic_sft \
  --model mamba2-1.3b \
  --model_path ./checkpoints/1.3b_mixed_wiki/biomamba_cpt_singledoc_mamba2-1.3b/best_model \
  --task both --epochs 3 --lr 2e-5 \
  --max_train_samples 20000 \
  --batch_size 16 --accum 2 \
  --output_dir ./checkpoints/mimic_sft_1.3b_cpt

# 2.7b CPT+SFT — max batch=8, use 8, accum=4
echo "$(date) Starting 2.7b CPT+SFT..."
python -m ft_biomamba.run_mimic_sft \
  --model mamba2-2.7b \
  --model_path ./checkpoints/2.7b_mixed_wiki/biomamba_cpt_singledoc_mamba2-2.7b/best_model \
  --task both --epochs 3 --lr 1e-5 \
  --max_train_samples 20000 \
  --batch_size 8 --accum 4 \
  --output_dir ./checkpoints/mimic_sft_2.7b_cpt

echo "$(date) === All CPT+SFT training complete! ==="
