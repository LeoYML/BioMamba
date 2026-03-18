#!/bin/bash
# Refine around v10 sweet spot (lr=5e-6, 3ep, wd=0.03, warmup=0.15, 65_35, ratio=0.70)
# Target: F1 > 0.819
cd "$(dirname "$0")/../.."

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

CPT_PATH="./checkpoints/1.3b_mixed_wiki/biomamba_cpt_singledoc_mamba2-1.3b/best_model"

# v12: 50:50 balanced data, same lr/ep as v10
echo "====== v12: 50:50 balanced ======"
python scripts/training/finetune_pubmedqa_sft.py \
  --model_name mamba2-1.3b \
  --model_path "$CPT_PATH" \
  --batch_size 4 --accumulation_steps 8 \
  --lr 5e-6 --num_epochs 3 \
  --max_length 1024 --warmup_ratio 0.15 --weight_decay 0.03 --max_grad_norm 1.0 \
  --output_dir ./checkpoints/1.3b_sft_v12 \
  --log_dir ./runs \
  --gpu_id 0 --num_proc 1 --logging_steps 5 --eval_steps 10 --save_steps 50 \
  --bf16 --reprocess_data \
  --mix_bioasq --bioasq_data_path ./data/bioasq_combined_50_50 --bioasq_split train --bioasq_train_ratio 0.70

# v13: v10 + 4 epochs (more training)
echo "====== v13: 4 epochs ======"
python scripts/training/finetune_pubmedqa_sft.py \
  --model_name mamba2-1.3b \
  --model_path "$CPT_PATH" \
  --batch_size 4 --accumulation_steps 8 \
  --lr 5e-6 --num_epochs 4 \
  --max_length 1024 --warmup_ratio 0.15 --weight_decay 0.03 --max_grad_norm 1.0 \
  --output_dir ./checkpoints/1.3b_sft_v13 \
  --log_dir ./runs \
  --gpu_id 0 --num_proc 1 --logging_steps 5 --eval_steps 10 --save_steps 50 \
  --bf16 --reprocess_data \
  --mix_bioasq --bioasq_data_path ./data/bioasq_combined_65_35 --bioasq_split train --bioasq_train_ratio 0.70

# v14: slightly higher lr (7e-6), 3ep, 65_35 ratio=0.70
echo "====== v14: lr=7e-6 ======"
python scripts/training/finetune_pubmedqa_sft.py \
  --model_name mamba2-1.3b \
  --model_path "$CPT_PATH" \
  --batch_size 4 --accumulation_steps 8 \
  --lr 7e-6 --num_epochs 3 \
  --max_length 1024 --warmup_ratio 0.15 --weight_decay 0.03 --max_grad_norm 1.0 \
  --output_dir ./checkpoints/1.3b_sft_v14 \
  --log_dir ./runs \
  --gpu_id 0 --num_proc 1 --logging_steps 5 --eval_steps 10 --save_steps 50 \
  --bf16 --reprocess_data \
  --mix_bioasq --bioasq_data_path ./data/bioasq_combined_65_35 --bioasq_split train --bioasq_train_ratio 0.70

# v15: 50:50 balanced + 4 epochs + slightly higher lr
echo "====== v15: 50:50 + 4ep + lr=7e-6 ======"
python scripts/training/finetune_pubmedqa_sft.py \
  --model_name mamba2-1.3b \
  --model_path "$CPT_PATH" \
  --batch_size 4 --accumulation_steps 8 \
  --lr 7e-6 --num_epochs 4 \
  --max_length 1024 --warmup_ratio 0.15 --weight_decay 0.03 --max_grad_norm 1.0 \
  --output_dir ./checkpoints/1.3b_sft_v15 \
  --log_dir ./runs \
  --gpu_id 0 --num_proc 1 --logging_steps 5 --eval_steps 10 --save_steps 50 \
  --bf16 --reprocess_data \
  --mix_bioasq --bioasq_data_path ./data/bioasq_combined_50_50 --bioasq_split train --bioasq_train_ratio 0.70

echo "====== Training done! ======"

# Evaluate
for v in v12 v13 v14 v15; do
  CKPT=$(find ./checkpoints/1.3b_sft_${v} -name "best_model" -type d | head -1)
  if [ -n "$CKPT" ]; then
    echo "--- BioASQ: ${v} ---"
    python scripts/evaluation/evaluate_bioasq.py \
      --model_path "$CKPT" \
      --model_type mamba2 \
      --dataset bioasq \
      --data_path ./data/bioasq_test \
      --split test \
      --output_dir ./evaluation_results/1.3b_f1_${v} \
      --gpu_id 0
  fi
done

echo "====== All done! ======"
