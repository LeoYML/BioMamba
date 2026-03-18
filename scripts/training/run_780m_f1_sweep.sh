#!/bin/bash
# 780M F1 optimization: apply v14 recipe (65_35, ratio=0.70, 3ep, wd=0.03)
# Current best: v1 Acc=79.27% F1=0.781 (lr=2e-5, 5ep, wd=0.01, bioasq_65_35 ratio=0.65)
cd "$(dirname "$0")/../.."

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

CPT_PATH="./checkpoints/780m_mixed_wiki/biomamba_cpt_singledoc_mamba2-780m/best_model"

# v2: v14 recipe (lr=1e-5, 3ep, wd=0.03, 65_35, ratio=0.70)
echo "====== 780M v2: v14 recipe lr=1e-5 ======"
python scripts/training/finetune_pubmedqa_sft.py \
  --model_name mamba2-780m \
  --model_path "$CPT_PATH" \
  --batch_size 8 --accumulation_steps 8 \
  --lr 1e-5 --num_epochs 3 \
  --max_length 1024 --warmup_ratio 0.15 --weight_decay 0.03 --max_grad_norm 1.0 \
  --output_dir ./checkpoints/780m_sft_v2 \
  --log_dir ./runs \
  --gpu_id 0 --num_proc 1 --logging_steps 5 --eval_steps 10 --save_steps 50 \
  --bf16 --reprocess_data \
  --mix_bioasq --bioasq_data_path ./data/bioasq_combined_65_35 --bioasq_split train --bioasq_train_ratio 0.70

# v3: higher lr (2e-5, 3ep, wd=0.03, 65_35, ratio=0.70)
echo "====== 780M v3: lr=2e-5 3ep ======"
python scripts/training/finetune_pubmedqa_sft.py \
  --model_name mamba2-780m \
  --model_path "$CPT_PATH" \
  --batch_size 8 --accumulation_steps 8 \
  --lr 2e-5 --num_epochs 3 \
  --max_length 1024 --warmup_ratio 0.15 --weight_decay 0.03 --max_grad_norm 1.0 \
  --output_dir ./checkpoints/780m_sft_v3 \
  --log_dir ./runs \
  --gpu_id 0 --num_proc 1 --logging_steps 5 --eval_steps 10 --save_steps 50 \
  --bf16 --reprocess_data \
  --mix_bioasq --bioasq_data_path ./data/bioasq_combined_65_35 --bioasq_split train --bioasq_train_ratio 0.70

# v4: lower lr (5e-6, 3ep, wd=0.03, 65_35, ratio=0.70)
echo "====== 780M v4: lr=5e-6 3ep ======"
python scripts/training/finetune_pubmedqa_sft.py \
  --model_name mamba2-780m \
  --model_path "$CPT_PATH" \
  --batch_size 8 --accumulation_steps 8 \
  --lr 5e-6 --num_epochs 3 \
  --max_length 1024 --warmup_ratio 0.15 --weight_decay 0.03 --max_grad_norm 1.0 \
  --output_dir ./checkpoints/780m_sft_v4 \
  --log_dir ./runs \
  --gpu_id 0 --num_proc 1 --logging_steps 5 --eval_steps 10 --save_steps 50 \
  --bf16 --reprocess_data \
  --mix_bioasq --bioasq_data_path ./data/bioasq_combined_65_35 --bioasq_split train --bioasq_train_ratio 0.70

echo "====== Training done! ======"

# Evaluate
for v in v2 v3 v4; do
  CKPT=$(find ./checkpoints/780m_sft_${v} -name "best_model" -type d | head -1)
  if [ -n "$CKPT" ]; then
    echo "--- BioASQ: 780M ${v} ---"
    python scripts/evaluation/evaluate_bioasq.py \
      --model_path "$CKPT" \
      --model_type mamba2 \
      --dataset bioasq \
      --data_path ./data/bioasq_test \
      --split test \
      --output_dir ./evaluation_results/780m_f1_${v} \
      --gpu_id 0
  fi
done

echo "====== All done! ======"
