"""
Visualize model predictions on validation samples.

Shows predicted depth vs ground truth for multiple validation images.

Usage:
    python visualize_predictions.py --checkpoint checkpoints/depth_jepa_vit_small.pt --num_samples 3
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

from src.data.depth_ds import DepthDataset, collate_depth
from src.models.depth_model import DepthViT


def denormalize_image(img_tensor):
    """Reverse ImageNet normalization for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def depth_to_colormap(depth_tensor, cmap='magma', vmin=None, vmax=None):
    """Convert depth tensor to colored visualization."""
    depth = depth_tensor.squeeze().cpu().float().numpy()  # Convert to float32 for numpy
    if vmin is None:
        vmin = depth.min()
    if vmax is None:
        vmax = depth.max()
    # Normalize to [0, 1] for colormap
    depth_norm = (depth - vmin) / (vmax - vmin + 1e-8)
    depth_norm = np.clip(depth_norm, 0, 1)
    # Apply colormap
    cm = plt.get_cmap(cmap)
    colored = cm(depth_norm)[:, :, :3]  # Remove alpha channel
    return colored, vmin, vmax


def compute_metrics(pred, target):
    """Compute depth metrics for a single sample."""
    pred = pred.squeeze()
    target = target.squeeze()
    
    eps = 1e-6
    pred = pred.clamp(min=eps)
    target = target.clamp(min=eps)
    
    # Absolute relative error
    abs_rel = torch.mean(torch.abs(pred - target) / target).item()
    
    # RMSE
    rmse = torch.sqrt(torch.mean((pred - target) ** 2)).item()
    
    # Delta thresholds
    ratio = torch.max(pred / target, target / pred)
    delta1 = (ratio < 1.25).float().mean().item()
    delta2 = (ratio < 1.25 ** 2).float().mean().item()
    delta3 = (ratio < 1.25 ** 3).float().mean().item()
    
    return {
        'abs_rel': abs_rel,
        'rmse': rmse,
        'delta1': delta1,
        'delta2': delta2,
        'delta3': delta3,
    }


def visualize_validation_predictions(
    model, 
    dataset, 
    device, 
    num_samples=3, 
    save_path="validation_predictions.png"
):
    """
    Visualize predictions on validation samples.
    
    Creates a figure with:
    - Row per sample: RGB | Ground Truth Depth | Predicted Depth | Error Map
    """
    model.eval()
    
    # Select random samples
    np.random.seed(42)
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    all_metrics = []
    
    with torch.no_grad():
        for row, idx in enumerate(indices):
            # Get sample (single view mode for validation)
            img, depth_gt = dataset[idx]
            
            # Add batch dimension and move to device
            img_batch = img.unsqueeze(0).to(device, dtype=torch.bfloat16)
            depth_gt = depth_gt.to(device, dtype=torch.bfloat16)
            
            # Predict
            pred_depth = model(img_batch, return_embedding=False)
            pred_depth = pred_depth.squeeze(0)  # Remove batch dim
            
            # Compute metrics
            metrics = compute_metrics(pred_depth.float(), depth_gt.float())
            all_metrics.append(metrics)
            
            # Get common depth range for consistent coloring
            vmin = min(depth_gt.min().item(), pred_depth.min().item())
            vmax = max(depth_gt.max().item(), pred_depth.max().item())
            
            # Column 0: RGB input
            img_vis = denormalize_image(img)
            axes[row, 0].imshow(img_vis)
            axes[row, 0].set_title(f"Sample {idx}: RGB Input", fontsize=10)
            axes[row, 0].axis('off')
            
            # Column 1: Ground truth depth
            gt_vis, _, _ = depth_to_colormap(depth_gt, vmin=vmin, vmax=vmax)
            axes[row, 1].imshow(gt_vis)
            axes[row, 1].set_title("Ground Truth Depth", fontsize=10)
            axes[row, 1].axis('off')
            
            # Column 2: Predicted depth
            pred_vis, _, _ = depth_to_colormap(pred_depth, vmin=vmin, vmax=vmax)
            axes[row, 2].imshow(pred_vis)
            axes[row, 2].set_title(f"Predicted Depth\nδ1={metrics['delta1']:.3f}", fontsize=10)
            axes[row, 2].axis('off')
            
            # Column 3: Error map (absolute difference)
            error = torch.abs(pred_depth.float() - depth_gt.float()).squeeze().cpu().numpy()
            im = axes[row, 3].imshow(error, cmap='hot', vmin=0, vmax=0.3)
            axes[row, 3].set_title(f"Abs Error\nAbsRel={metrics['abs_rel']:.3f}, RMSE={metrics['rmse']:.3f}", fontsize=10)
            axes[row, 3].axis('off')
            plt.colorbar(im, ax=axes[row, 3], fraction=0.046, pad=0.04)
    
    # Add overall title with average metrics
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
    fig.suptitle(
        f"JEPA Depth Estimation - Validation Results\n"
        f"Avg: AbsRel={avg_metrics['abs_rel']:.4f} | RMSE={avg_metrics['rmse']:.4f} | "
        f"δ1={avg_metrics['delta1']:.4f} | δ2={avg_metrics['delta2']:.4f} | δ3={avg_metrics['delta3']:.4f}",
        fontsize=12, fontweight='bold'
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved visualization to {save_path}")
    print(f"\nMetrics per sample:")
    for i, (idx, m) in enumerate(zip(indices, all_metrics)):
        print(f"  Sample {idx}: AbsRel={m['abs_rel']:.4f}, RMSE={m['rmse']:.4f}, "
              f"δ1={m['delta1']:.4f}, δ2={m['delta2']:.4f}, δ3={m['delta3']:.4f}")
    print(f"\nAverage:")
    print(f"  AbsRel={avg_metrics['abs_rel']:.4f}, RMSE={avg_metrics['rmse']:.4f}, "
          f"δ1={avg_metrics['delta1']:.4f}, δ2={avg_metrics['delta2']:.4f}, δ3={avg_metrics['delta3']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Visualize depth predictions on validation set")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/depth_jepa_vit_small.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--num_samples", type=int, default=3,
                       help="Number of validation samples to visualize")
    parser.add_argument("--output", type=str, default="validation_predictions.png",
                       help="Output image path")
    parser.add_argument("--neighborhoods", type=str, default=None,
                       help="Comma-separated neighborhood IDs (default: all)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda or cpu)")
    args = parser.parse_args()
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return
    
    # Parse neighborhoods
    neighborhoods = None
    if args.neighborhoods:
        neighborhoods = [int(n) for n in args.neighborhoods.split(",")]
    
    # Setup device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load validation dataset (single view mode)
    print("Loading validation dataset...")
    val_dataset = DepthDataset(
        split="val",
        global_img_size=224,
        local_img_size=96,
        neighborhoods=neighborhoods,
        multi_view=False,  # Single view for validation
    )
    
    if len(val_dataset) == 0:
        print("Error: No validation samples found!")
        return
    
    print(f"Validation dataset has {len(val_dataset)} samples")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Get model config from checkpoint if available
    model_name = "vit_small_patch16_224.augreg_in21k"
    if "args" in checkpoint:
        model_name = checkpoint["args"].get("model", model_name)
    
    model = DepthViT(
        model_name=model_name,
        img_size=224,
        pretrained=False,  # Don't load pretrained, we'll load checkpoint
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(torch.bfloat16)
    model.eval()
    
    print(f"Model loaded. Checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    if "val_metrics" in checkpoint:
        m = checkpoint["val_metrics"]
        print(f"Checkpoint validation metrics: δ1={m.get('delta1', 'N/A'):.4f}")
    
    # Visualize
    visualize_validation_predictions(
        model=model,
        dataset=val_dataset,
        device=device,
        num_samples=args.num_samples,
        save_path=args.output,
    )


if __name__ == "__main__":
    main()
