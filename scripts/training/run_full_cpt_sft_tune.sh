#!/bin/bash
# CPT+SFT Final Tuning — All 5 sizes
#
# Strategy (based on v2 results):
# - lr=1e-5 beats lr=2e-5 on ROUGE despite worse val_loss
# - 30k samples per task (vs 20k in base+SFT) for more discharge data
# - 5 epochs with patience=5 early stopping
#
# v2 results showed lr=1e-5 drastically improved 1.3b generation:
#   1.3b v2: C-R1 8.16 (vs 7.93 base), D-R1 10.27 (vs 10.01 base)
#   Only D-R2 still -0.04 below base
cd "$(dirname "$0")/../.."

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================"
echo "  CPT+SFT v3 — All 5 Sizes"
echo "  lr=1e-5, data=30k, epochs=5"
echo "============================================"

echo ""
echo "=== [1/5] 130m CPT+SFT v3 ==="
python -m ft_biomamba.run_mimic_sft \
  --model mamba2-130m \
  --model_path ./checkpoints/mixed_wiki/biomamba_cpt_singledoc_mamba2-130m/best_model \
  --task both --epochs 5 --lr 1e-5 \
  --max_train_samples 30000 \
  --batch_size 8 --accum 4 \
  --warmup_ratio 0.1 \
  --output_dir ./checkpoints/mimic_sft_130m_cpt_v3

echo ""
echo "=== [2/5] 370m CPT+SFT v3 ==="
python -m ft_biomamba.run_mimic_sft \
  --model mamba2-370m \
  --model_path ./checkpoints/370m_mixed_wiki/biomamba_cpt_singledoc_mamba2-370m/best_model \
  --task both --epochs 5 --lr 1e-5 \
  --max_train_samples 30000 \
  --batch_size 32 --accum 1 \
  --warmup_ratio 0.1 \
  --output_dir ./checkpoints/mimic_sft_370m_cpt_v3

echo ""
echo "=== [3/5] 780m CPT+SFT v3 ==="
python -m ft_biomamba.run_mimic_sft \
  --model mamba2-780m \
  --model_path ./checkpoints/780m_mixed_wiki/biomamba_cpt_singledoc_mamba2-780m/best_model \
  --task both --epochs 5 --lr 1e-5 \
  --max_train_samples 30000 \
  --batch_size 16 --accum 2 \
  --warmup_ratio 0.1 \
  --output_dir ./checkpoints/mimic_sft_780m_cpt_v3

echo ""
echo "=== [4/5] 1.3b CPT+SFT v3 ==="
python -m ft_biomamba.run_mimic_sft \
  --model mamba2-1.3b \
  --model_path ./checkpoints/1.3b_mixed_wiki/biomamba_cpt_singledoc_mamba2-1.3b/best_model \
  --task both --epochs 5 --lr 1e-5 \
  --max_train_samples 30000 \
  --batch_size 16 --accum 2 \
  --warmup_ratio 0.1 \
  --output_dir ./checkpoints/mimic_sft_1.3b_cpt_v3

echo ""
echo "=== [5/5] 2.7b CPT+SFT v3 ==="
python -m ft_biomamba.run_mimic_sft \
  --model mamba2-2.7b \
  --model_path ./checkpoints/2.7b_mixed_wiki/biomamba_cpt_singledoc_mamba2-2.7b/best_model \
  --task both --epochs 5 --lr 5e-6 \
  --max_train_samples 30000 \
  --batch_size 4 --accum 8 \
  --warmup_ratio 0.1 \
  --output_dir ./checkpoints/mimic_sft_2.7b_cpt_v3

echo ""
echo "============================================"
echo "  Stage 2: Evaluate ALL v3 models"
echo "============================================"

python scripts/evaluation/run_all_mimic_eval.py \
  "130m-CPT+SFT-v3" "130m-base+SFT" \
  "370m-CPT+SFT-v3" "370m-base+SFT" \
  "780m-CPT+SFT-v3" "780m-base+SFT" \
  "1.3b-CPT+SFT-v3" "1.3b-base+SFT" \
  "2.7b-CPT+SFT-v3" "2.7b-base+SFT"

echo ""
echo "============================================"
echo "  All done! Check evaluation_results/mimic_v2/"
echo "============================================"
