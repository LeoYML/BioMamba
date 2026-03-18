#!/bin/bash
# 1.3b CPT+SFT v5 — Fix D-R2 gap from v2
#
# v2 (lr=1e-5, 20k, 5ep): C-R1=8.16✓ C-R2=3.48✓ D-R1=10.27✓ D-R2=3.68✗(-0.02)
# v4 (lr=7.5e-6, 20k, 8ep): C-R1=7.92✗ D-R2=3.70✓ but C-R2=3.24✗
#
# v5: lr=9e-6, 20k, 6 epochs — slightly lower than v2's 1e-5 to improve D-R2
# With more warmup (0.12) for smoother optimization
cd "$(dirname "$0")/../.."

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================"
echo "  1.3b CPT+SFT v5 (fix D-R2)"
echo "  lr=9e-6, data=20k, epochs=6"
echo "============================================"

python -m ft_biomamba.run_mimic_sft \
  --model mamba2-1.3b \
  --model_path ./checkpoints/1.3b_mixed_wiki/biomamba_cpt_singledoc_mamba2-1.3b/best_model \
  --task both --epochs 6 --lr 9e-6 \
  --max_train_samples 20000 \
  --batch_size 16 --accum 2 \
  --warmup_ratio 0.12 \
  --output_dir ./checkpoints/mimic_sft_1.3b_cpt_v5

echo ""
echo "============================================"
echo "  Evaluate v5"
echo "============================================"

python scripts/evaluation/run_all_mimic_eval.py "1.3b-CPT+SFT-v5"

echo ""
echo "All done!"
