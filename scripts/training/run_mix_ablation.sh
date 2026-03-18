#!/bin/bash
# Ablation: 130m CPT with various C4/Wiki mixing ratios
# Compare PubMed PPL against baseline (10%C4+10%Wiki, val_loss=2.111)

cd "$(dirname "$0")/../.."

source ./.venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/BioMamba

echo "=========================================="
echo "Step 1: Prepare mixed datasets"
echo "=========================================="
python scripts/data_preparation/prepare_mix_ablation_data.py

# Common training args
COMMON="--model mamba2-130m --epochs 3 --lr 5e-6 --batch_size 32 --accum 8 --label_smoothing 0.0 --layer_lr_decay 0.9 --gpu_id 0 --seed 42"

echo ""
echo "=========================================="
echo "Exp A: PubMed + 5% C4"
echo "=========================================="
python scripts/training/run_cpt_singledoc.py $COMMON \
    --data_dir ./data/pubmed_mix_c4_only_5pct \
    --output_dir ./checkpoints/mix_c4only5

sleep 10

echo ""
echo "=========================================="
echo "Exp B: PubMed + 20% C4"
echo "=========================================="
python scripts/training/run_cpt_singledoc.py $COMMON \
    --data_dir ./data/pubmed_mix_c4_only_20pct \
    --output_dir ./checkpoints/mix_c4only20

sleep 10

echo ""
echo "=========================================="
echo "Exp C: PubMed + 30% C4"
echo "=========================================="
python scripts/training/run_cpt_singledoc.py $COMMON \
    --data_dir ./data/pubmed_mix_c4_only_30pct \
    --output_dir ./checkpoints/mix_c4only30

sleep 10

echo ""
echo "=========================================="
echo "Exp D: PubMed + 5%C4 + 5%Wiki"
echo "=========================================="
python scripts/training/run_cpt_singledoc.py $COMMON \
    --data_dir ./data/pubmed_mix_c4_5_wiki5 \
    --output_dir ./checkpoints/mix_c45_wiki5 \

sleep 10

echo ""
echo "=========================================="
echo "Exp E: PubMed + 20%C4 + 10%Wiki"
echo "=========================================="
python scripts/training/run_cpt_singledoc.py $COMMON \
    --data_dir ./data/pubmed_mix_c420_wiki10 \
    --output_dir ./checkpoints/mix_c420_wiki10

sleep 10

echo ""
echo "=========================================="
echo "Step 7: Evaluate all models on PubMed PPL"
echo "=========================================="
python scripts/evaluation/eval_mix_ablation.py

echo ""
echo "All mix ablation experiments complete!"
