#!/bin/bash
# 1.3B CPT+SFT F1 optimization sweep
# Current best: v7b F1=0.787 (yes-F1=0.867, no-F1=0.706)
# Target: F1 > 0.80, improve no-F1
cd "$(dirname "$0")/../.."

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

CPT_PATH="./checkpoints/1.3b_mixed_wiki/biomamba_cpt_singledoc_mamba2-1.3b/best_model"

# v8: 2.7B's config (lr=2e-5, 5ep, wd=0.01, bioasq_65_35 ratio=0.65) — balanced data
echo "====== 1.3B CPT v8: 2.7B config ======"
python scripts/training/finetune_pubmedqa_sft.py \
  --model_name mamba2-1.3b \
  --model_path "$CPT_PATH" \
  --batch_size 4 --accumulation_steps 8 \
  --lr 2e-5 --num_epochs 5 \
  --max_length 1024 --warmup_ratio 0.1 --weight_decay 0.01 --max_grad_norm 1.0 \
  --output_dir ./checkpoints/1.3b_sft_v8 \
  --log_dir ./runs \
  --gpu_id 0 --num_proc 1 --logging_steps 5 --eval_steps 10 --save_steps 50 \
  --bf16 --reprocess_data \
  --mix_bioasq --bioasq_data_path ./data/bioasq_combined_65_35 --bioasq_split train --bioasq_train_ratio 0.65

# v9: moderate (lr=1e-5, 3ep, wd=0.01, bioasq_65_35 ratio=0.65)
echo "====== 1.3B CPT v9: moderate balanced ======"
python scripts/training/finetune_pubmedqa_sft.py \
  --model_name mamba2-1.3b \
  --model_path "$CPT_PATH" \
  --batch_size 4 --accumulation_steps 8 \
  --lr 1e-5 --num_epochs 3 \
  --max_length 1024 --warmup_ratio 0.1 --weight_decay 0.01 --max_grad_norm 1.0 \
  --output_dir ./checkpoints/1.3b_sft_v9 \
  --log_dir ./runs \
  --gpu_id 0 --num_proc 1 --logging_steps 5 --eval_steps 10 --save_steps 50 \
  --bf16 --reprocess_data \
  --mix_bioasq --bioasq_data_path ./data/bioasq_combined_65_35 --bioasq_split train --bioasq_train_ratio 0.65

# v10: sweet spot (lr=5e-6, 3ep, wd=0.03, bioasq_65_35 ratio=0.70)
echo "====== 1.3B CPT v10: sweet spot ======"
python scripts/training/finetune_pubmedqa_sft.py \
  --model_name mamba2-1.3b \
  --model_path "$CPT_PATH" \
  --batch_size 4 --accumulation_steps 8 \
  --lr 5e-6 --num_epochs 3 \
  --max_length 1024 --warmup_ratio 0.15 --weight_decay 0.03 --max_grad_norm 1.0 \
  --output_dir ./checkpoints/1.3b_sft_v10 \
  --log_dir ./runs \
  --gpu_id 0 --num_proc 1 --logging_steps 5 --eval_steps 10 --save_steps 50 \
  --bf16 --reprocess_data \
  --mix_bioasq --bioasq_data_path ./data/bioasq_combined_65_35 --bioasq_split train --bioasq_train_ratio 0.70

# v11: more bioasq balanced (lr=8e-6, 3ep, wd=0.03, bioasq_all ratio=0.75)
echo "====== 1.3B CPT v11: more data balanced ======"
python scripts/training/finetune_pubmedqa_sft.py \
  --model_name mamba2-1.3b \
  --model_path "$CPT_PATH" \
  --batch_size 4 --accumulation_steps 8 \
  --lr 8e-6 --num_epochs 3 \
  --max_length 1024 --warmup_ratio 0.1 --weight_decay 0.03 --max_grad_norm 1.0 \
  --output_dir ./checkpoints/1.3b_sft_v11 \
  --log_dir ./runs \
  --gpu_id 0 --num_proc 1 --logging_steps 5 --eval_steps 10 --save_steps 50 \
  --bf16 --reprocess_data \
  --mix_bioasq --bioasq_data_path ./data/bioasq_combined_all --bioasq_split train --bioasq_train_ratio 0.75

echo "====== All training done! ======"

# Evaluate all on BioASQ
echo "====== Evaluating BioASQ ======"
for v in v8 v9 v10 v11; do
  CKPT=$(find ./checkpoints/1.3b_sft_${v} -name "best_model" -type d | head -1)
  if [ -n "$CKPT" ]; then
    echo "--- BioASQ: 1.3B ${v} ($CKPT) ---"
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

echo "====== All evaluations done! ======"
