#!/bin/bash

# Convenience wrapper for full SFT mode
# You can still override any variable via env, e.g.:
#   MODEL_PATH=... MAX_LENGTH=1024 bash run_sft_pubmedqa_full.sh

cd "$(dirname "$0")/../.."

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TRAINING_MODE="${TRAINING_MODE:-full}"
export USE_WANDB="${USE_WANDB:-true}"
export WANDB_PROJECT="${WANDB_PROJECT:-biomamba2-sft-full}"

bash "$SCRIPT_DIR/run_sft_pubmedqa.sh"
