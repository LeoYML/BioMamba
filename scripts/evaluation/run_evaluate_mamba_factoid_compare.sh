#!/bin/bash

################################################################################
# Extract BioASQ factoid test set and compare:
# 1) Original Mamba2 model
# 2) Bio cpretrain Mamba2 model
################################################################################

cd "$(dirname "$0")/../.."

set -e

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
    if [ -x ".venv/bin/python" ]; then
        PYTHON_BIN=".venv/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi

GOLDEN_FILES=(
    "13B1_golden.json"
    "13B2_golden.json"
    "13B3_golden.json"
    "13B4_golden.json"
)

FACTOID_DATA_PATH="${FACTOID_DATA_PATH:-./data/bioasq_factoid_test}"
SPLIT="${SPLIT:-test}"
MAX_SAMPLES="${MAX_SAMPLES:-}"            # empty = full factoid set
MAX_LENGTH="${MAX_LENGTH:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-20}"    # factoid often needs >1 token
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.9}"
DO_SAMPLE="${DO_SAMPLE:-false}"
GPU_ID="${GPU_ID:-0}"

# Model paths
ORIG_MODEL_PATH="${ORIG_MODEL_PATH:-state-spaces/mamba2-130m}"
CPRETRAIN_MODEL_PATH="${CPRETRAIN_MODEL_PATH:-./checkpoints/biomamba2_mamba2-130m_20260209_002521/best_model}"

# Output
OUTPUT_BASE="${OUTPUT_BASE:-./evaluation_results/mamba2_factoid_compare}"
ORIG_OUTPUT_DIR="${ORIG_OUTPUT_DIR:-${OUTPUT_BASE}/original}"
CPRETRAIN_OUTPUT_DIR="${CPRETRAIN_OUTPUT_DIR:-${OUTPUT_BASE}/cpretrain}"

echo "=========================================="
echo "BioASQ Factoid: Mamba Original vs CPretrain"
echo "=========================================="
echo "Python bin:        $PYTHON_BIN"
echo "Factoid data path: $FACTOID_DATA_PATH"
echo "Original model:    $ORIG_MODEL_PATH"
echo "CPretrain model:   $CPRETRAIN_MODEL_PATH"
echo "Max samples:       ${MAX_SAMPLES:-All}"
echo "=========================================="
echo ""

for f in "${GOLDEN_FILES[@]}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing golden file: $f"
        exit 1
    fi
done

if [ ! -d "$CPRETRAIN_MODEL_PATH" ]; then
    echo "ERROR: CPretrain model path not found: $CPRETRAIN_MODEL_PATH"
    exit 1
fi

echo "Step 1/3: Extracting factoid test set..."
"$PYTHON_BIN" scripts/data_preparation/convert_bioasq_golden_to_dataset.py \
    --input_files "${GOLDEN_FILES[@]}" \
    --output_dir "$FACTOID_DATA_PATH" \
    --question_type factoid

run_eval() {
    local model_path="$1"
    local output_dir="$2"

    local cmd=(
        "$PYTHON_BIN" scripts/evaluation/evaluate_bioasq.py
        --model_type mamba2
        --model_path "$model_path"
        --dataset_name bioasq
        --data_path "$FACTOID_DATA_PATH"
        --split "$SPLIT"
        --max_length "$MAX_LENGTH"
        --max_new_tokens "$MAX_NEW_TOKENS"
        --temperature "$TEMPERATURE"
        --top_p "$TOP_P"
        --gpu_id "$GPU_ID"
        --output_dir "$output_dir"
        --save_predictions
    )

    if [ -n "$MAX_SAMPLES" ]; then
        cmd+=(--max_samples "$MAX_SAMPLES")
    fi

    if [ "$DO_SAMPLE" = true ]; then
        cmd+=(--do_sample)
    fi

    "${cmd[@]}"
}

echo ""
echo "Step 2/3: Evaluating original Mamba model..."
run_eval "$ORIG_MODEL_PATH" "$ORIG_OUTPUT_DIR"

echo ""
echo "Step 3/3: Evaluating cpretrain Mamba model..."
run_eval "$CPRETRAIN_MODEL_PATH" "$CPRETRAIN_OUTPUT_DIR"

echo ""
echo "Done."
echo "Original results:  $ORIG_OUTPUT_DIR"
echo "CPretrain results: $CPRETRAIN_OUTPUT_DIR"
