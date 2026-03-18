#!/bin/bash
# 780M PubMedQA optimization sweep
# Current: v1 PubMedQA=66.5%, v3 PubMedQA=64.5% (BioASQ-optimized)
# Target: improve PubMedQA while keeping BioASQ reasonable
cd "$(dirname "$0")/../.."

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

CPT_PATH="./checkpoints/780m_mixed_wiki/biomamba_cpt_singledoc_mamba2-780m/best_model"

# v5: v1 recipe + wd=0.03 warmup=0.15 (keep ratio=0.65, 5ep — PubMedQA-friendly)
echo "====== 780M v5: v1+wd=0.03 ======"
python scripts/training/finetune_pubmedqa_sft.py \
  --model_name mamba2-780m \
  --model_path "$CPT_PATH" \
  --batch_size 8 --accumulation_steps 8 \
  --lr 2e-5 --num_epochs 5 \
  --max_length 1024 --warmup_ratio 0.15 --weight_decay 0.03 --max_grad_norm 1.0 \
  --output_dir ./checkpoints/780m_sft_v5 \
  --log_dir ./runs \
  --gpu_id 0 --num_proc 1 --logging_steps 5 --eval_steps 10 --save_steps 50 \
  --bf16 --reprocess_data \
  --mix_bioasq --bioasq_data_path ./data/bioasq_combined_65_35 --bioasq_split train --bioasq_train_ratio 0.65

# v6: lower ratio=0.50 (more PubMedQA weight), lr=2e-5, 3ep
echo "====== 780M v6: ratio=0.50 ======"
python scripts/training/finetune_pubmedqa_sft.py \
  --model_name mamba2-780m \
  --model_path "$CPT_PATH" \
  --batch_size 8 --accumulation_steps 8 \
  --lr 2e-5 --num_epochs 3 \
  --max_length 1024 --warmup_ratio 0.15 --weight_decay 0.03 --max_grad_norm 1.0 \
  --output_dir ./checkpoints/780m_sft_v6 \
  --log_dir ./runs \
  --gpu_id 0 --num_proc 1 --logging_steps 5 --eval_steps 10 --save_steps 50 \
  --bf16 --reprocess_data \
  --mix_bioasq --bioasq_data_path ./data/bioasq_combined_65_35 --bioasq_split train --bioasq_train_ratio 0.50

# v7: ratio=0.60, 5ep, wd=0.01 (closer to 2.7B's PubMedQA-winning recipe)
echo "====== 780M v7: 2.7B recipe ======"
python scripts/training/finetune_pubmedqa_sft.py \
  --model_name mamba2-780m \
  --model_path "$CPT_PATH" \
  --batch_size 8 --accumulation_steps 8 \
  --lr 2e-5 --num_epochs 5 \
  --max_length 1024 --warmup_ratio 0.1 --weight_decay 0.01 --max_grad_norm 1.0 \
  --output_dir ./checkpoints/780m_sft_v7 \
  --log_dir ./runs \
  --gpu_id 0 --num_proc 1 --logging_steps 5 --eval_steps 10 --save_steps 50 \
  --bf16 --reprocess_data \
  --mix_bioasq --bioasq_data_path ./data/bioasq_combined_all --bioasq_split train --bioasq_train_ratio 0.60

echo "====== Training done! ======"

# Evaluate BioASQ + PubMedQA
for v in v5 v6 v7; do
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

    echo "--- PubMedQA: 780M ${v} ---"
    python scripts/evaluation/eval_pubmedqa_generative.py \
      --models "$CKPT" \
      --strategies logprob \
      --prompt sft
  fi
done

echo "====== All done! ======"
