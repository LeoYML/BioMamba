#!/bin/bash
# Comprehensive CPT+SFT retuning for ALL sizes
# Strategy: Lower LR than base+SFT (CPT models are already adapted, need gentler fine-tuning)
# Goal: ALL non-PPL metrics must beat base+SFT
#
# Current gaps (CPT+SFT - base+SFT):
#   130m: C-R1 -0.01, C-R2 -0.02 (tiny)         → lr=1.5e-5, epochs=5
#   370m: C-R1 -0.03, C-R2 -0.01, C-RL -0.04    → lr=1.5e-5, epochs=5
#   780m: D-R1 -0.13, D-R2 -0.03, D-RL -0.08    → lr=1e-5, epochs=5 (v2 DONE)
#   1.3b: D-R2 -0.17, D-RL -0.03                 → lr=1e-5, epochs=5 (v2 running)
#   2.7b: C-R1 -0.01, C-R2 -0.09                 → lr=5e-6, epochs=5
cd "$(dirname "$0")/../.."

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== [1/3] 130m CPT+SFT v3 (lr=1.5e-5, epochs=5) ==="
python -m ft_biomamba.run_mimic_sft \
  --model mamba2-130m \
  --model_path ./checkpoints/mixed_wiki/biomamba_cpt_singledoc_mamba2-130m/best_model \
  --task both --epochs 5 --lr 1.5e-5 \
  --max_train_samples 20000 \
  --batch_size 8 --accum 4 \
  --warmup_ratio 0.1 \
  --output_dir ./checkpoints/mimic_sft_130m_cpt_v3

echo "=== [2/3] 370m CPT+SFT v3 (lr=1.5e-5, epochs=5) ==="
python -m ft_biomamba.run_mimic_sft \
  --model mamba2-370m \
  --model_path ./checkpoints/370m_mixed_wiki/biomamba_cpt_singledoc_mamba2-370m/best_model \
  --task both --epochs 5 --lr 1.5e-5 \
  --max_train_samples 20000 \
  --batch_size 32 --accum 1 \
  --warmup_ratio 0.1 \
  --output_dir ./checkpoints/mimic_sft_370m_cpt_v3

echo "=== [3/3] 2.7b CPT+SFT v2 (lr=5e-6, epochs=5) ==="
python -m ft_biomamba.run_mimic_sft \
  --model mamba2-2.7b \
  --model_path ./checkpoints/2.7b_mixed_wiki/biomamba_cpt_singledoc_mamba2-2.7b/best_model \
  --task both --epochs 5 --lr 5e-6 \
  --max_train_samples 20000 \
  --batch_size 4 --accum 8 \
  --warmup_ratio 0.1 \
  --output_dir ./checkpoints/mimic_sft_2.7b_cpt_v2

echo "=== All CPT+SFT retraining complete ==="
