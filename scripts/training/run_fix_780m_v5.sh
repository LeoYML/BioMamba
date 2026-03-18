#!/bin/bash
# 780m CPT+SFT v5 — Split-the-difference LR
#
# v3 (lr=1e-5, 30k): C-R1=7.92✓ D-R1=8.78✗(-0.08)
# v4 (lr=5e-6, 20k): C-R1=7.85✗(-0.05) D-R1=9.13✓
# v5: lr=7.5e-6, 20k, 6 epochs — midpoint between v3 and v4
cd "$(dirname "$0")/../.."

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================"
echo "  780m CPT+SFT v5 (balance C-R1 & D-R1)"
echo "  lr=7.5e-6, data=20k, epochs=6"
echo "============================================"

python -m ft_biomamba.run_mimic_sft \
  --model mamba2-780m \
  --model_path ./checkpoints/780m_mixed_wiki/biomamba_cpt_singledoc_mamba2-780m/best_model \
  --task both --epochs 6 --lr 7.5e-6 \
  --max_train_samples 20000 \
  --batch_size 16 --accum 2 \
  --warmup_ratio 0.12 \
  --output_dir ./checkpoints/mimic_sft_780m_cpt_v5

echo ""
echo "============================================"
echo "  Evaluate v5"
echo "============================================"

python scripts/evaluation/run_all_mimic_eval.py "780m-CPT+SFT-v5"

echo ""
echo "All done!"
