#!/bin/bash
# Evaluate all PPL baselines on BioASQ and PubMedQA
cd "$(dirname "$0")/../.."

set -euo pipefail

# BioASQ evaluations for baselines we haven't tested yet
echo "====== BioGPT-Large-PubMedQA (BioASQ) ======"
python scripts/evaluation/evaluate_bioasq.py \
  --model_path microsoft/BioGPT-Large-PubMedQA \
  --model_type biogpt \
  --dataset bioasq \
  --data_path ./data/bioasq_test \
  --split test \
  --output_dir ./evaluation_results/baseline_biogpt_large_pqa \
  --gpu_id 0

echo "====== BioMedLM (BioASQ) ======"
python scripts/evaluation/evaluate_bioasq.py \
  --model_path stanford-crfm/BioMedLM \
  --model_type auto \
  --dataset bioasq \
  --data_path ./data/bioasq_test \
  --split test \
  --output_dir ./evaluation_results/baseline_biomedlm \
  --gpu_id 0

echo "====== Meditron3-Gemma2-2B (BioASQ) ======"
python scripts/evaluation/evaluate_bioasq.py \
  --model_path OpenMeditron/Meditron3-Gemma2-2B \
  --model_type auto \
  --dataset bioasq \
  --data_path ./data/bioasq_test \
  --split test \
  --output_dir ./evaluation_results/baseline_meditron3 \
  --gpu_id 0

echo "====== Bio-Medical-Llama-3.2-1B (BioASQ) ======"
python scripts/evaluation/evaluate_bioasq.py \
  --model_path ContactDoctor/Bio-Medical-Llama-3-2-1B-CoT-012025 \
  --model_type auto \
  --dataset bioasq \
  --data_path ./data/bioasq_test \
  --split test \
  --output_dir ./evaluation_results/baseline_llama_bio \
  --gpu_id 0

echo "====== All BioASQ baselines done! ======"
