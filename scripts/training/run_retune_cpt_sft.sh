#!/bin/bash
# Retune 780m and 1.3b CPT+SFT with lower lr to preserve CPT knowledge
# 780m CPT+SFT v1: lr=2e-5, epochs=3 → D-R1=8.73 (worse than base+SFT 8.86)
# 1.3b CPT+SFT v1: lr=2e-5, epochs=3 → D-R2=3.55 (worse than base+SFT 3.72)
# Fix: lr=1e-5, epochs=5 (rely on early stopping patience=5)
cd "$(dirname "$0")/../.."

set -e

# 780m CPT+SFT v2 — lr=1e-5, epochs=5, batch=16, accum=2
echo "$(date) Starting 780m CPT+SFT retune (lr=1e-5, epochs=5)..."
python -m ft_biomamba.run_mimic_sft \
  --model mamba2-780m \
  --model_path ./checkpoints/780m_mixed_wiki/biomamba_cpt_singledoc_mamba2-780m/best_model \
  --task both --epochs 5 --lr 1e-5 \
  --max_train_samples 20000 \
  --batch_size 16 --accum 2 \
  --warmup_ratio 0.1 \
  --output_dir ./checkpoints/mimic_sft_780m_cpt_v2

echo "$(date) 780m CPT+SFT retune complete!"

# 1.3b CPT+SFT v2 — lr=1e-5, epochs=5, batch=16, accum=2
echo "$(date) Starting 1.3b CPT+SFT retune (lr=1e-5, epochs=5)..."
python -m ft_biomamba.run_mimic_sft \
  --model mamba2-1.3b \
  --model_path ./checkpoints/1.3b_mixed_wiki/biomamba_cpt_singledoc_mamba2-1.3b/best_model \
  --task both --epochs 5 --lr 1e-5 \
  --max_train_samples 20000 \
  --batch_size 16 --accum 2 \
  --warmup_ratio 0.1 \
  --output_dir ./checkpoints/mimic_sft_1.3b_cpt_v2

echo "$(date) 1.3b CPT+SFT retune complete!"
echo "$(date) === All retuning complete! ==="
