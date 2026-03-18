#!/bin/bash
# CPT for Mamba2-2.7b using mixed data (PubMed + 10% C4 + 10% Wikipedia)
# 4x H100 80GB DDP training with gradient checkpointing
# Per-GPU: BS=16, Accum=4 → Effective batch: 16 * 4 * 4 = 256
cd "$(dirname "$0")/../.."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

source ./.venv/bin/activate

torchrun --nproc_per_node=4 --master_port=29500 \
    scripts/training/run_cpt_singledoc.py \
    --model mamba2-2.7b \
    --data_dir ./data/pubmed_mixed_10pct_general_10pct_wiki \
    --epochs 3 \
    --lr 5e-6 \
    --batch_size 6 \
    --accum 8 \
    --label_smoothing 0.0 \
    --layer_lr_decay 0.95 \
    --output_dir ./checkpoints/2.7b_mixed_wiki \
    --seed 42 \
    --grad_ckpt
