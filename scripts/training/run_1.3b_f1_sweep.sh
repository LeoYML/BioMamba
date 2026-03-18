#!/bin/bash
# 1.3B F1 optimization sweep on NoCPT base
# Target: BioASQ F1 between 780M (0.781) and 2.7B (0.890)
cd "$(dirname "$0")/../.."

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# v1: 2.7B's exact config on NoCPT 1.3B (lr=2e-5, 5ep, bioasq_65_35, ratio=0.65)
echo "====== 1.3B NoCPT SFT v1: 2.7B config ======"
python scripts/training/finetune_pubmedqa_sft.py \
  --model_name mamba2-1.3b \
  --model_path state-spaces/mamba2-1.3b \
  --batch_size 4 --accumulation_steps 8 \
  --lr 2e-5 --num_epochs 5 \
  --max_length 1024 --warmup_ratio 0.1 --weight_decay 0.01 --max_grad_norm 1.0 \
  --output_dir ./checkpoints/1.3b_f1_v1 \
  --log_dir ./runs \
  --gpu_id 0 --num_proc 1 --logging_steps 5 --eval_steps 10 --save_steps 50 \
  --bf16 --reprocess_data \
  --mix_bioasq --bioasq_data_path ./data/bioasq_combined_65_35 --bioasq_split train --bioasq_train_ratio 0.65

# v2: moderate lr, 3ep, balanced data (bioasq_65_35, ratio=0.65)
echo "====== 1.3B NoCPT SFT v2: moderate ======"
python scripts/training/finetune_pubmedqa_sft.py \
  --model_name mamba2-1.3b \
  --model_path state-spaces/mamba2-1.3b \
  --batch_size 4 --accumulation_steps 8 \
  --lr 1e-5 --num_epochs 3 \
  --max_length 1024 --warmup_ratio 0.1 --weight_decay 0.01 --max_grad_norm 1.0 \
  --output_dir ./checkpoints/1.3b_f1_v2 \
  --log_dir ./runs \
  --gpu_id 0 --num_proc 1 --logging_steps 5 --eval_steps 10 --save_steps 50 \
  --bf16 --reprocess_data \
  --mix_bioasq --bioasq_data_path ./data/bioasq_combined_65_35 --bioasq_split train --bioasq_train_ratio 0.65

# v3: low lr, 2ep, more bioasq (bioasq_all, ratio=0.80)
echo "====== 1.3B NoCPT SFT v3: more bioasq ======"
python scripts/training/finetune_pubmedqa_sft.py \
  --model_name mamba2-1.3b \
  --model_path state-spaces/mamba2-1.3b \
  --batch_size 4 --accumulation_steps 8 \
  --lr 5e-6 --num_epochs 2 \
  --max_length 1024 --warmup_ratio 0.15 --weight_decay 0.05 --max_grad_norm 1.0 \
  --output_dir ./checkpoints/1.3b_f1_v3 \
  --log_dir ./runs \
  --gpu_id 0 --num_proc 1 --logging_steps 5 --eval_steps 10 --save_steps 50 \
  --bf16 --reprocess_data \
  --mix_bioasq --bioasq_data_path ./data/bioasq_combined_all --bioasq_split train --bioasq_train_ratio 0.80

# v4: moderate lr, 3ep, balanced + more data (bioasq_all, ratio=0.70)
echo "====== 1.3B NoCPT SFT v4: balanced+more ======"
python scripts/training/finetune_pubmedqa_sft.py \
  --model_name mamba2-1.3b \
  --model_path state-spaces/mamba2-1.3b \
  --batch_size 4 --accumulation_steps 8 \
  --lr 1e-5 --num_epochs 3 \
  --max_length 1024 --warmup_ratio 0.1 --weight_decay 0.03 --max_grad_norm 1.0 \
  --output_dir ./checkpoints/1.3b_f1_v4 \
  --log_dir ./runs \
  --gpu_id 0 --num_proc 1 --logging_steps 5 --eval_steps 10 --save_steps 50 \
  --bf16 --reprocess_data \
  --mix_bioasq --bioasq_data_path ./data/bioasq_combined_all --bioasq_split train --bioasq_train_ratio 0.70

# v5: CPT base + balanced data (test if CPT + balanced data > NoCPT)
echo "====== 1.3B CPT SFT v5: CPT+balanced ======"
python scripts/training/finetune_pubmedqa_sft.py \
  --model_name mamba2-1.3b \
  --model_path ./checkpoints/1.3b_mixed_wiki/biomamba_cpt_singledoc_mamba2-1.3b/best_model \
  --batch_size 4 --accumulation_steps 8 \
  --lr 5e-6 --num_epochs 2 \
  --max_length 1024 --warmup_ratio 0.15 --weight_decay 0.05 --max_grad_norm 1.0 \
  --output_dir ./checkpoints/1.3b_f1_v5 \
  --log_dir ./runs \
  --gpu_id 0 --num_proc 1 --logging_steps 5 --eval_steps 10 --save_steps 50 \
  --bf16 --reprocess_data \
  --mix_bioasq --bioasq_data_path ./data/bioasq_combined_65_35 --bioasq_split train --bioasq_train_ratio 0.65

echo "====== All 1.3B F1 sweep training done! ======"

# Evaluate all on BioASQ
echo "====== Evaluating BioASQ ======"
for v in v1 v2 v3 v4 v5; do
  CKPT=$(find ./checkpoints/1.3b_f1_${v} -name "best_model" -type d | head -1)
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

echo "====== All 1.3B F1 sweep evaluations done! ======"
