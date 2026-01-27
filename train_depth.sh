#!/bin/bash
#SBATCH --job-name=depth
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:rtx_pro_6000:1

# Create logs directory
mkdir -p logs
mkdir -p checkpoints

# Activate environment (adjust as needed)
# source ~/.bashrc
# conda activate jepa

# ============================================
# JEPA Training (Recommended - Multi-view with SIGReg)
# ============================================
# Same loss function logic as run_JEPA.py and loss.py:
# - 2 global views (224x224) + 4 local views (96x96)
# - LeJEPA loss: center prediction + SIGReg regularization
# - Depth supervision on all views

python train_depth_jepa.py \
    --epochs 50 \
    --bs 16 \
    --lr 1e-4 \
    --global_img_size 224 \
    --local_img_size 96 \
    --model vit_small_patch16_224.augreg_in21k \
    --num_workers 8 \
    --grad_accum 1 \
    --V_global 2 \
    --V_local 4 \
    --lamb 0.05 \
    --depth_weight 1.0 \
    --jepa_weight 0.5 \
    --wandb \
    --save_path checkpoints/depth_jepa_vit_small.pt

# ============================================
# Alternative: Simple depth training (no multi-view)
# ============================================
# python train_depth.py \
#     --epochs 50 \
#     --bs 32 \
#     --lr 1e-4 \
#     --img_size 224 \
#     --model vit_small_patch16_224.augreg_in21k \
#     --num_workers 8 \
#     --sigreg_weight 0.1 \
#     --wandb \
#     --save_path checkpoints/depth_vit_small.pt

# ============================================
# Alternative: vit_base for better quality (slower)
# ============================================
# python train_depth_jepa.py \
#     --epochs 30 \
#     --bs 8 \
#     --lr 5e-5 \
#     --global_img_size 224 \
#     --local_img_size 96 \
#     --model vit_base_patch16_224.dino \
#     --num_workers 8 \
#     --grad_accum 2 \
#     --V_global 2 \
#     --V_local 4 \
#     --lamb 0.05 \
#     --wandb \
#     --save_path checkpoints/depth_jepa_vit_base.pt
