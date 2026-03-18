#!/bin/bash
# Fix remaining gaps after v3:
#   780m: D-R1 -0.08 below base+SFT (v3 lr=1e-5, 30k data)
#   1.3b: v2 (lr=1e-5, 20k) was better than v3, only D-R2 -0.04 gap
#
# Key insight: 30k data HURT 780m D-R1 and 1.3b performance
# → Revert to 20k data but use even lower LR for these two sizes
#
# 780m v4: lr=5e-6, 20k, 8 epochs — gentler training for D-R1
# 1.3b v4: lr=7.5e-6, 20k, 8 epochs — slightly lower than v2's 1e-5
cd "$(dirname "$0")/../.."

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================"
echo "  [1/2] 780m CPT+SFT v4 (fix D-R1)"
echo "  lr=5e-6, data=20k, epochs=8"
echo "============================================"

python -m ft_biomamba.run_mimic_sft \
  --model mamba2-780m \
  --model_path ./checkpoints/780m_mixed_wiki/biomamba_cpt_singledoc_mamba2-780m/best_model \
  --task both --epochs 8 --lr 5e-6 \
  --max_train_samples 20000 \
  --batch_size 16 --accum 2 \
  --warmup_ratio 0.15 \
  --output_dir ./checkpoints/mimic_sft_780m_cpt_v4

echo ""
echo "============================================"
echo "  [2/2] 1.3b CPT+SFT v4 (fix D-R2)"
echo "  lr=7.5e-6, data=20k, epochs=8"
echo "============================================"

python -m ft_biomamba.run_mimic_sft \
  --model mamba2-1.3b \
  --model_path ./checkpoints/1.3b_mixed_wiki/biomamba_cpt_singledoc_mamba2-1.3b/best_model \
  --task both --epochs 8 --lr 7.5e-6 \
  --max_train_samples 20000 \
  --batch_size 16 --accum 2 \
  --warmup_ratio 0.15 \
  --output_dir ./checkpoints/mimic_sft_1.3b_cpt_v4

echo ""
echo "============================================"
echo "  Evaluate v4 models"
echo "============================================"

python scripts/evaluation/run_all_mimic_eval.py \
  "780m-CPT+SFT-v4" "780m-base+SFT" \
  "1.3b-CPT+SFT-v4" "1.3b-base+SFT"

echo ""
echo "All done!"
