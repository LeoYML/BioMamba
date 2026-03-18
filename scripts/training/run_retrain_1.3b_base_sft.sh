#!/bin/bash
# Retrain 1.3b base+SFT with lr=3e-5 (slightly higher than original 2e-5)
cd "$(dirname "$0")/../.."

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Backup old results
cp ./evaluation_results/mimic_v2/1.3b-base_SFT.json ./evaluation_results/mimic_v2/1.3b-base_SFT_orig.json 2>/dev/null || true

echo "============================================"
echo "  1.3b base+SFT retrain (lr=5e-5)"
echo "============================================"

python -m ft_biomamba.run_mimic_sft \
  --model mamba2-1.3b \
  --model_path state-spaces/mamba2-1.3b \
  --task both --epochs 3 --lr 5e-5 \
  --max_train_samples 20000 \
  --batch_size 16 --accum 2 \
  --output_dir ./checkpoints/mimic_sft_1.3b

echo ""
echo "============================================"
echo "  Evaluate retrained base+SFT"
echo "============================================"

python scripts/evaluation/run_all_mimic_eval.py "1.3b-base+SFT"

echo ""
echo "All done!"
