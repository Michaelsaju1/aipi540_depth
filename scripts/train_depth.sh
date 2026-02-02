#!/bin/bash
#SBATCH --job-name=depth
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a6000:1


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

# ./run_inf.sh  python src/visualization/visualize_predictions.py --checkpoint checkpoints/depth_jepa_vit_small.pt --num_samples 3 --output validation_predictions.png 

# ============================================
# Alternative: Simple depth training (no multi-view)
# ============================================
# python src/training/train_depth.py \
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


# Fail fast
set -e

# Activate environment
source /home/users/aho13/jepa_tests/env/bin/activate

# Optional: debugging
echo "Running on $(hostname)"
echo "Running on partition: $SLURM_JOB_PARTITION"
echo "Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"

nvidia-smi
which python
python --version

# Run training
export HYDRA_FULL_ERROR=1

# Memory-efficient training configuration:
./run_inf.sh python src/training/train_depth_jepa.py \
    --epochs 30 \
    --bs 128 \
    --lr 5e-5 \
    --global_img_size 224 \
    --local_img_size 96 \
    --prefetch_factor 4 \
    --model vit_small_patch16_224.augreg_in21k \
    --num_workers 6 \
    --grad_accum 1 \
    --V_global 2 \
    --V_local 4 \
    --lamb 0.05 \
    --wandb \
    --save_path checkpoints/depth_jepa_vit_small2.pt