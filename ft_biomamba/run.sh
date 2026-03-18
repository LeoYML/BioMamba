#!/bin/bash
# ===========================================================================
# BioMamba Fine-tuning Pipeline - One-Click Run Script
#
# Pipeline: CPT (PubMed-MEDLINE) -> SFT (PubMedQA+BioASQ) -> Eval (BioASQ)
#
# Usage:
#   bash ft_biomamba/run.sh                    # full pipeline (default 130m)
#   bash ft_biomamba/run.sh --model mamba2-370m # larger model
#   bash ft_biomamba/run.sh --skip_cpt         # skip CPT stage
#   bash ft_biomamba/run.sh --eval_only        # eval only (needs sft_checkpoint)
# ===========================================================================
set -euo pipefail

# Ensure we're in the project root
cd "$(dirname "$0")/.."

# ---- Defaults (override via env vars or CLI args) ----
MODEL="${MODEL:-mamba2-130m}"
GPU_ID="${GPU_ID:-0}"
DATA_DIR="${DATA_DIR:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints}"
SEED="${SEED:-42}"

# CPT settings
CPT_EPOCHS="${CPT_EPOCHS:-3}"
CPT_LR="${CPT_LR:-5e-6}"
CPT_MAX_LEN="${CPT_MAX_LEN:-1024}"
CPT_ACCUM="${CPT_ACCUM:-8}"

# SFT settings
SFT_EPOCHS="${SFT_EPOCHS:-5}"
SFT_LR="${SFT_LR:-2e-5}"
SFT_MAX_LEN="${SFT_MAX_LEN:-1024}"
SFT_BATCH="${SFT_BATCH:-8}"
SFT_ACCUM="${SFT_ACCUM:-4}"

# Eval settings
EVAL_DATASET="${EVAL_DATASET:-bioasq}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-./data/bioasq_test}"

# BioASQ training data for SFT mixing
BIOASQ_TRAIN_PATH="${BIOASQ_TRAIN_PATH:-./data/bioasq_13b_yesno_train}"

# ---- Parse CLI args ----
EXTRA_ARGS=()
SKIP_CPT=""
EVAL_ONLY=""

for arg in "$@"; do
    case "$arg" in
        --skip_cpt) SKIP_CPT="--skip_cpt" ;;
        --eval_only) EVAL_ONLY="1" ;;
        --model=*) MODEL="${arg#*=}" ;;
        *) EXTRA_ARGS+=("$arg") ;;
    esac
done

# ---- Activate virtualenv if present ----
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "============================================================"
echo "  BioMamba Fine-tuning Pipeline"
echo "============================================================"
echo "  Model:     ${MODEL}"
echo "  GPU:       ${GPU_ID}"
echo "  Data dir:  ${DATA_DIR}"
echo "  Output:    ${OUTPUT_DIR}"
echo "============================================================"

if [ -n "${EVAL_ONLY}" ]; then
    echo "  Mode: Evaluation only"
    echo "============================================================"
    python -m ft_biomamba.run_pipeline \
        --model "${MODEL}" \
        --skip_cpt --skip_sft \
        --eval_dataset "${EVAL_DATASET}" \
        --eval_data_path "${EVAL_DATA_PATH}" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --gpu_id "${GPU_ID}" \
        --seed "${SEED}" \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
else
    echo "  Mode: Full pipeline (CPT -> SFT -> Eval)"
    echo "============================================================"

    # Build BioASQ mixing args
    BIOASQ_ARGS=""
    if [ -d "${BIOASQ_TRAIN_PATH}" ]; then
        BIOASQ_ARGS="--mix_bioasq --bioasq_data_path ${BIOASQ_TRAIN_PATH}"
        echo "  BioASQ mix: enabled (${BIOASQ_TRAIN_PATH})"
    else
        BIOASQ_ARGS="--no_mix_bioasq"
        echo "  BioASQ mix: disabled (path not found)"
    fi

    python -m ft_biomamba.run_pipeline \
        --model "${MODEL}" \
        ${SKIP_CPT} \
        --cpt_epochs "${CPT_EPOCHS}" \
        --cpt_lr "${CPT_LR}" \
        --cpt_max_length "${CPT_MAX_LEN}" \
        --cpt_accum "${CPT_ACCUM}" \
        --sft_epochs "${SFT_EPOCHS}" \
        --sft_lr "${SFT_LR}" \
        --sft_max_length "${SFT_MAX_LEN}" \
        --sft_batch_size "${SFT_BATCH}" \
        --sft_accum "${SFT_ACCUM}" \
        ${BIOASQ_ARGS} \
        --eval_dataset "${EVAL_DATASET}" \
        --eval_data_path "${EVAL_DATA_PATH}" \
        --compare_base \
        --data_dir "${DATA_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --gpu_id "${GPU_ID}" \
        --seed "${SEED}" \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
fi

echo ""
echo "============================================================"
echo "  Pipeline finished!"
echo "============================================================"
