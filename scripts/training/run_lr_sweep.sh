#!/bin/bash

################################################################################
# LR Sweep Script
#
# Runs multiple pilot experiments with different learning rates to find the
# optimal LR for continued pretraining.
#
# Results will appear in wandb project "biomamba-pilot" with distinct run names
# so you can overlay the loss curves for direct comparison.
#
# Usage:
#   bash run_lr_sweep.sh              # Default: test 3e-6, 5e-6, 1e-5, 2e-5
#   bash run_lr_sweep.sh --steps 300  # Shorter runs for faster comparison
################################################################################

cd "$(dirname "$0")/../.."

set -e
source ./.venv/bin/activate

# Default sweep configuration
STEPS=500
EXTRA_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --steps)
            STEPS="$2"; shift 2 ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

# LR values to sweep
LR_VALUES=("3e-6" "5e-6" "1e-5" "2e-5")

echo ""
echo "=============================================="
echo "Learning Rate Sweep"
echo "=============================================="
echo ""
echo "LR values:  ${LR_VALUES[*]}"
echo "Steps each: $STEPS"
echo "Total runs: ${#LR_VALUES[@]}"
echo "Extra args: ${EXTRA_ARGS:-none}"
echo ""
echo "Results → wandb project: biomamba-pilot"
echo "=============================================="
echo ""

for LR in "${LR_VALUES[@]}"; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Running pilot with LR=$LR"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # Create descriptive run name for wandb
    RUN_NAME="lr_${LR}_steps_${STEPS}"
    
    bash scripts/training/run_training_pilot.sh \
        --lr "$LR" \
        --steps "$STEPS" \
        --run_name "$RUN_NAME" \
        $EXTRA_ARGS \
    || echo "WARNING: Run with LR=$LR failed, continuing with next..."
    
    echo ""
    echo "Completed LR=$LR"
    echo ""
done

echo ""
echo "=============================================="
echo "LR Sweep Complete!"
echo "=============================================="
echo ""
echo "Compare results in wandb:"
echo "  https://wandb.ai → biomamba-pilot project"
echo ""
echo "Look for:"
echo "  - Which LR shows steepest loss decrease?"
echo "  - Which LR has stable grad_norm (no spikes)?"
echo "  - Which LR gives lowest eval perplexity at step $STEPS?"
echo ""
echo "Typical patterns:"
echo "  LR too low  → loss barely moves, grad_norm very small"
echo "  LR too high → loss spikes or NaN, grad_norm explodes"
echo "  LR optimal  → smooth loss decrease, moderate grad_norm"
echo "=============================================="
