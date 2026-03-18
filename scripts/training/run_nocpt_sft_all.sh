#!/bin/bash
# SFT on base models (no CPT) for all sizes, using same hyperparams as best CPT+SFT versions
cd "$(dirname "$0")/../.."

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# 370M: same config as 370M v7 (best balanced)
echo "====== 370M NoCPT SFT ======"
python scripts/training/finetune_pubmedqa_sft.py \
  --model_name mamba2-370m \
  --model_path state-spaces/mamba2-370m \
  --batch_size 16 --accumulation_steps 4 \
  --lr 3e-5 --num_epochs 2 \
  --max_length 1024 --warmup_ratio 0.1 --weight_decay 0.01 --max_grad_norm 1.0 \
  --output_dir ./checkpoints/370m_nocpt_sft \
  --log_dir ./runs \
  --gpu_id 0 --num_proc 1 --logging_steps 5 --eval_steps 10 --save_steps 50 \
  --bf16 --reprocess_data \
  --mix_bioasq --bioasq_data_path ./data/bioasq_combined_all --bioasq_split train --bioasq_train_ratio 0.70

# 780M: same config as 780M v1
echo "====== 780M NoCPT SFT ======"
python scripts/training/finetune_pubmedqa_sft.py \
  --model_name mamba2-780m \
  --model_path state-spaces/mamba2-780m \
  --batch_size 16 --accumulation_steps 4 \
  --lr 2e-5 --num_epochs 5 \
  --max_length 1024 --warmup_ratio 0.1 --weight_decay 0.01 --max_grad_norm 1.0 \
  --output_dir ./checkpoints/780m_nocpt_sft \
  --log_dir ./runs \
  --gpu_id 0 --num_proc 1 --logging_steps 5 --eval_steps 10 --save_steps 50 \
  --bf16 --reprocess_data \
  --mix_bioasq --bioasq_data_path ./data/bioasq_combined_65_35 --bioasq_split train --bioasq_train_ratio 0.65

# 1.3B: same config as 1.3B v7b (best BioASQ)
echo "====== 1.3B NoCPT SFT ======"
python scripts/training/finetune_pubmedqa_sft.py \
  --model_name mamba2-1.3b \
  --model_path state-spaces/mamba2-1.3b \
  --batch_size 4 --accumulation_steps 8 \
  --lr 5e-6 --num_epochs 2 \
  --max_length 1024 --warmup_ratio 0.15 --weight_decay 0.05 --max_grad_norm 1.0 \
  --output_dir ./checkpoints/1.3b_nocpt_sft \
  --log_dir ./runs \
  --gpu_id 0 --num_proc 1 --logging_steps 5 --eval_steps 10 --save_steps 50 \
  --bf16 --reprocess_data \
  --mix_bioasq --bioasq_data_path ./data/bioasq_combined_all --bioasq_split train --bioasq_train_ratio 0.90

# 2.7B: same config as 2.7B v1
echo "====== 2.7B NoCPT SFT ======"
python scripts/training/finetune_pubmedqa_sft.py \
  --model_name mamba2-2.7b \
  --model_path state-spaces/mamba2-2.7b \
  --batch_size 4 --accumulation_steps 8 \
  --lr 2e-5 --num_epochs 5 \
  --max_length 1024 --warmup_ratio 0.1 --weight_decay 0.01 --max_grad_norm 1.0 \
  --output_dir ./checkpoints/2.7b_nocpt_sft \
  --log_dir ./runs \
  --gpu_id 0 --num_proc 1 --logging_steps 5 --eval_steps 10 --save_steps 50 \
  --bf16 --reprocess_data \
  --mix_bioasq --bioasq_data_path ./data/bioasq_combined_65_35 --bioasq_split train --bioasq_train_ratio 0.65

echo "====== All NoCPT SFT training done! ======"

# Evaluate all on BioASQ
echo "====== Evaluating BioASQ ======"
for size in 370m 780m 1.3b 2.7b; do
  CKPT=$(find ./checkpoints/${size}_nocpt_sft -name "best_model" -type d | head -1)
  if [ -n "$CKPT" ]; then
    echo "--- BioASQ: $size ($CKPT) ---"
    python scripts/evaluation/evaluate_bioasq.py \
      --model_path "$CKPT" \
      --model_type mamba2 \
      --dataset bioasq \
      --data_path ./data/bioasq_test \
      --split test \
      --output_dir ./evaluation_results/nocpt_sft_${size} \
      --gpu_id 0
  fi
done

# Evaluate all on PubMedQA (logprob)
echo "====== Evaluating PubMedQA ======"
MODELS=""
for size in 370m 780m 1.3b 2.7b; do
  CKPT=$(find ./checkpoints/${size}_nocpt_sft -name "best_model" -type d | head -1)
  if [ -n "$CKPT" ]; then
    MODELS="$MODELS $CKPT"
  fi
done
python scripts/evaluation/eval_pubmedqa_generative.py --models $MODELS --strategies logprob --prompt sft

echo "====== All NoCPT evaluations done! ======"
