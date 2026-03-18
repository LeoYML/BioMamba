#!/bin/bash
# Ablation v2: More C4+Wiki mixing ratios + pure PubMed baseline
# All use Mamba2-130m, 3 epochs, same hyperparams as baseline
# Then evaluate all ablation models on PubMed/Wiki/C4

cd "$(dirname "$0")/../.."

source ./.venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/BioMamba

echo "=========================================="
echo "Step 1: Prepare datasets"
echo "=========================================="
python scripts/data_preparation/prepare_mix_ablation_v2.py

COMMON="--model mamba2-130m --epochs 3 --lr 5e-6 --batch_size 32 --accum 8 --label_smoothing 0.0 --layer_lr_decay 0.9 --gpu_id 0 --seed 42"

# --- Exp 1: Pure PubMed (0% C4, 0% Wiki) ---
if [ ! -d "./checkpoints/pubmed_only/biomamba_cpt_singledoc_mamba2-130m/best_model" ]; then
    echo ""
    echo "=========================================="
    echo "Exp 1: Pure PubMed (0%C4 + 0%Wiki)"
    echo "=========================================="
    python scripts/training/run_cpt_singledoc.py $COMMON \
        --data_dir ./data/pubmed_mix_pubmed_only \
        --output_dir ./checkpoints/pubmed_only
    sleep 5
else
    echo "SKIP Exp 1: checkpoints/pubmed_only already exists"
fi

# --- Exp 2: 5%C4 + 5%Wiki ---
if [ ! -d "./checkpoints/mix_c45_wiki5/biomamba_cpt_singledoc_mamba2-130m/best_model" ]; then
    echo ""
    echo "=========================================="
    echo "Exp 2: 5%C4 + 5%Wiki"
    echo "=========================================="
    python scripts/training/run_cpt_singledoc.py $COMMON \
        --data_dir ./data/pubmed_mix_c45_wiki5 \
        --output_dir ./checkpoints/mix_c45_wiki5
    sleep 5
else
    echo "SKIP Exp 2: checkpoints/mix_c45_wiki5 already exists"
fi

# --- Exp 3: 5%C4 + 10%Wiki ---
if [ ! -d "./checkpoints/mix_c45_wiki10/biomamba_cpt_singledoc_mamba2-130m/best_model" ]; then
    echo ""
    echo "=========================================="
    echo "Exp 3: 5%C4 + 10%Wiki"
    echo "=========================================="
    python scripts/training/run_cpt_singledoc.py $COMMON \
        --data_dir ./data/pubmed_mix_c45_wiki10 \
        --output_dir ./checkpoints/mix_c45_wiki10
    sleep 5
else
    echo "SKIP Exp 3: checkpoints/mix_c45_wiki10 already exists"
fi

# --- Exp 4: 10%C4 + 5%Wiki ---
if [ ! -d "./checkpoints/mix_c410_wiki5/biomamba_cpt_singledoc_mamba2-130m/best_model" ]; then
    echo ""
    echo "=========================================="
    echo "Exp 4: 10%C4 + 5%Wiki"
    echo "=========================================="
    python scripts/training/run_cpt_singledoc.py $COMMON \
        --data_dir ./data/pubmed_mix_c410_wiki5 \
        --output_dir ./checkpoints/mix_c410_wiki5
    sleep 5
else
    echo "SKIP Exp 4: checkpoints/mix_c410_wiki5 already exists"
fi

# --- Exp 5: 10%C4 + 20%Wiki ---
if [ ! -d "./checkpoints/mix_c410_wiki20/biomamba_cpt_singledoc_mamba2-130m/best_model" ]; then
    echo ""
    echo "=========================================="
    echo "Exp 5: 10%C4 + 20%Wiki"
    echo "=========================================="
    python scripts/training/run_cpt_singledoc.py $COMMON \
        --data_dir ./data/pubmed_mix_c410_wiki20 \
        --output_dir ./checkpoints/mix_c410_wiki20
    sleep 5
else
    echo "SKIP Exp 5: checkpoints/mix_c410_wiki20 already exists"
fi

# --- Exp 6: 15%C4 + 15%Wiki ---
if [ ! -d "./checkpoints/mix_c415_wiki15/biomamba_cpt_singledoc_mamba2-130m/best_model" ]; then
    echo ""
    echo "=========================================="
    echo "Exp 6: 15%C4 + 15%Wiki"
    echo "=========================================="
    python scripts/training/run_cpt_singledoc.py $COMMON \
        --data_dir ./data/pubmed_mix_c415_wiki15 \
        --output_dir ./checkpoints/mix_c415_wiki15
    sleep 5
else
    echo "SKIP Exp 6: checkpoints/mix_c415_wiki15 already exists"
fi

# --- Exp 7: 20%C4 + 20%Wiki ---
if [ ! -d "./checkpoints/mix_c420_wiki20/biomamba_cpt_singledoc_mamba2-130m/best_model" ]; then
    echo ""
    echo "=========================================="
    echo "Exp 7: 20%C4 + 20%Wiki"
    echo "=========================================="
    python scripts/training/run_cpt_singledoc.py $COMMON \
        --data_dir ./data/pubmed_mix_c420_wiki20 \
        --output_dir ./checkpoints/mix_c420_wiki20
    sleep 5
else
    echo "SKIP Exp 7: checkpoints/mix_c420_wiki20 already exists"
fi

echo ""
echo "=========================================="
echo "Step 2: Evaluate all models on 3 domains"
echo "=========================================="
python scripts/evaluation/eval_ablation_all.py

echo ""
echo "All ablation v2 experiments complete!"
