#!/bin/bash
# Train and evaluate all model sizes on MIMIC
# 370m is already training, start from 780m

cd "$(dirname "$0")/../.."

set -e

# Wait for 370m to finish if still running
while pgrep -f "run_mimic_sft.*370m" > /dev/null 2>&1; do
    echo "$(date) Waiting for 370m training..."
    sleep 120
done
echo "$(date) 370m training done"

# 780m SFT
echo "$(date) Starting 780m SFT..."
python -m ft_biomamba.run_mimic_sft \
  --model mamba2-780m \
  --model_path state-spaces/mamba2-780m \
  --task both --epochs 3 --lr 2e-5 \
  --max_train_samples 20000 \
  --batch_size 4 --accum 8 \
  --output_dir ./checkpoints/mimic_sft_780m 2>&1 | grep -E "(Epoch.*train_loss|Best model|Trainer)"

# 1.3b SFT
echo "$(date) Starting 1.3b SFT..."
python -m ft_biomamba.run_mimic_sft \
  --model mamba2-1.3b \
  --model_path state-spaces/mamba2-1.3b \
  --task both --epochs 3 --lr 2e-5 \
  --max_train_samples 20000 \
  --batch_size 2 --accum 16 \
  --output_dir ./checkpoints/mimic_sft_1.3b 2>&1 | grep -E "(Epoch.*train_loss|Best model|Trainer)"

# 2.7b SFT
echo "$(date) Starting 2.7b SFT..."
python -m ft_biomamba.run_mimic_sft \
  --model mamba2-2.7b \
  --model_path state-spaces/mamba2-2.7b \
  --task both --epochs 3 --lr 1e-5 \
  --max_train_samples 20000 \
  --batch_size 1 --accum 32 \
  --output_dir ./checkpoints/mimic_sft_2.7b 2>&1 | grep -E "(Epoch.*train_loss|Best model|Trainer)"

echo "$(date) All training complete!"

# Now evaluate all models
echo "$(date) Starting evaluation of all models..."

# Evaluate 370m
python scripts/evaluation/run_all_mimic_eval.py "370m-base" "370m-base+SFT" 2>&1 | tail -10

# Evaluate 780m
python scripts/evaluation/run_all_mimic_eval.py "780m-base" "780m-base+SFT" 2>&1 | tail -10

# Evaluate 1.3b
python scripts/evaluation/run_all_mimic_eval.py "1.3b-base" "1.3b-base+SFT" 2>&1 | tail -10

# Evaluate 2.7b
python scripts/evaluation/run_all_mimic_eval.py "2.7b-base" "2.7b-base+SFT" 2>&1 | tail -10

echo "$(date) ALL DONE!"
