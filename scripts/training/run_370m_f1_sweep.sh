#!/bin/bash
# 370M F1 optimization: apply v14 recipe (65_35, ratio=0.70, 3ep, wd=0.03)
# Current best: v6 Acc=81.71% F1=0.756 (lr=3e-5, 2ep, bioasq_all ratio=0.80)
cd "$(dirname "$0")/../.."

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

CPT_PATH="./checkpoints/370m_mixed_wiki/biomamba_cpt_singledoc_mamba2-370m/best_model"

# v8: v14 recipe (lr=3e-5, 3ep, wd=0.03, 65_35, ratio=0.70)
echo "====== 370M v8: v14 recipe ======"
python scripts/training/finetune_pubmedqa_sft.py \
  --model_name mamba2-370m \
  --model_path "$CPT_PATH" \
  --batch_size 16 --accumulation_steps 4 \
  --lr 3e-5 --num_epochs 3 \
  --max_length 1024 --warmup_ratio 0.15 --weight_decay 0.03 --max_grad_norm 1.0 \
  --output_dir ./checkpoints/370m_sft_v8 \
  --log_dir ./runs \
  --gpu_id 0 --num_proc 1 --logging_steps 5 --eval_steps 10 --save_steps 50 \
  --bf16 --reprocess_data \
  --mix_bioasq --bioasq_data_path ./data/bioasq_combined_65_35 --bioasq_split train --bioasq_train_ratio 0.70

# v9: lower lr (2e-5, 3ep, wd=0.03, 65_35, ratio=0.70)
echo "====== 370M v9: lower lr ======"
python scripts/training/finetune_pubmedqa_sft.py \
  --model_name mamba2-370m \
  --model_path "$CPT_PATH" \
  --batch_size 16 --accumulation_steps 4 \
  --lr 2e-5 --num_epochs 3 \
  --max_length 1024 --warmup_ratio 0.15 --weight_decay 0.03 --max_grad_norm 1.0 \
  --output_dir ./checkpoints/370m_sft_v9 \
  --log_dir ./runs \
  --gpu_id 0 --num_proc 1 --logging_steps 5 --eval_steps 10 --save_steps 50 \
  --bf16 --reprocess_data \
  --mix_bioasq --bioasq_data_path ./data/bioasq_combined_65_35 --bioasq_split train --bioasq_train_ratio 0.70

# v10: higher lr (4e-5, 2ep, wd=0.03, 65_35, ratio=0.70)
echo "====== 370M v10: higher lr 2ep ======"
python scripts/training/finetune_pubmedqa_sft.py \
  --model_name mamba2-370m \
  --model_path "$CPT_PATH" \
  --batch_size 16 --accumulation_steps 4 \
  --lr 4e-5 --num_epochs 2 \
  --max_length 1024 --warmup_ratio 0.15 --weight_decay 0.03 --max_grad_norm 1.0 \
  --output_dir ./checkpoints/370m_sft_v10 \
  --log_dir ./runs \
  --gpu_id 0 --num_proc 1 --logging_steps 5 --eval_steps 10 --save_steps 50 \
  --bf16 --reprocess_data \
  --mix_bioasq --bioasq_data_path ./data/bioasq_combined_65_35 --bioasq_split train --bioasq_train_ratio 0.70

echo "====== Training done! ======"

# Evaluate
for v in v8 v9 v10; do
  CKPT=$(find ./checkpoints/370m_sft_${v} -name "best_model" -type d | head -1)
  if [ -n "$CKPT" ]; then
    echo "--- BioASQ: 370M ${v} ---"
    python scripts/evaluation/evaluate_bioasq.py \
      --model_path "$CKPT" \
      --model_type mamba2 \
      --dataset bioasq \
      --data_path ./data/bioasq_test \
      --split test \
      --output_dir ./evaluation_results/370m_f1_${v} \
      --gpu_id 0
  fi
done

echo "====== All done! ======"
