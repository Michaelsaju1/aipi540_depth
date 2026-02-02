#!/bin/bash
#SBATCH --job-name=depth
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4

# Create logs directory
mkdir -p logs

# Run visualization on 3 validation samples
python src/visualization/visualize_predictions.py \
    --checkpoint checkpoints/depth_jepa_vit_small.pt \
    --num_samples 3 \
    --output validation_predictions.png \
    --device cuda

echo "Visualization complete. Check validation_predictions.png"
