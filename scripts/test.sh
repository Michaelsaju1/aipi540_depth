#!/bin/bash
#SBATCH --job-name=test_depth
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1

# Create output directories
mkdir -p logs
mkdir -p test_results

echo "Starting Testing..."
echo "GPU: $SLURM_GPUS"
echo "Node: $(hostname)"

# Run test script
# Using run_inf.sh wrapper to ensure environment is set up correctly
./run_inf.sh python src/test/test.py \
    --model_path checkpoints/depth_jepa_vit_small2.pt \
    --save_dir test_results \
    --num_workers 4 \
    --img_size 1024

echo "Testing complete!"
