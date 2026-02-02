"""
Visualize the JEPA Depth Estimation Pipeline.

Shows:
1. Original RGB image and depth map
2. Multi-view crops (global + local) with their corresponding depth crops
3. (If model provided) Predicted depth vs ground truth

Usage:
    python visualize_pipeline.py --output pipeline_vis.png
    python visualize_pipeline.py --checkpoint checkpoints/depth_jepa.pt --output prediction_vis.png
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image
import argparse
import os

from src.data.depth_ds import DepthDataset


def denormalize_image(img_tensor):
    """Reverse ImageNet normalization for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor * std + mean
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def depth_to_colormap(depth_tensor, cmap='magma'):
    """Convert depth tensor to colored visualization."""
    depth = depth_tensor.squeeze().numpy()
    # Normalize to [0, 1] for colormap
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    # Apply colormap
    cm = plt.get_cmap(cmap)
    colored = cm(depth_norm)[:, :, :3]  # Remove alpha channel
    return colored


def visualize_dataset_sample(dataset, idx=0, save_path="pipeline_dataset.png"):
    """
    Visualize a single sample from the dataset showing multi-view generation.
    """
    # Get the raw image and depth first (before transforms)
    img_path, depth_path = dataset.samples[idx]
    
    # Load original images
    orig_img = Image.open(img_path).convert("RGB")
    orig_depth = Image.open(depth_path)
    orig_depth_arr = np.array(orig_depth, dtype=np.float32) / 65535.0
    
    # Get multi-view sample
    img_views, depth_views = dataset[idx]
    
    V_global = dataset.V_global
    V_local = dataset.V_local
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(4, max(V_global + V_local, 2) + 1, figure=fig, 
                          height_ratios=[1.5, 1, 1, 1])
    
    # Row 0: Original image and depth
    ax_orig_img = fig.add_subplot(gs[0, 0])
    ax_orig_img.imshow(orig_img)
    ax_orig_img.set_title(f"Original RGB\n{orig_img.size[0]}×{orig_img.size[1]}", fontsize=10)
    ax_orig_img.axis('off')
    
    ax_orig_depth = fig.add_subplot(gs[0, 1])
    im = ax_orig_depth.imshow(orig_depth_arr, cmap='magma')
    ax_orig_depth.set_title(f"Original Depth\n{orig_depth_arr.shape[1]}×{orig_depth_arr.shape[0]}", fontsize=10)
    ax_orig_depth.axis('off')
    plt.colorbar(im, ax=ax_orig_depth, fraction=0.046, pad=0.04)
    
    # Add pipeline description
    ax_text = fig.add_subplot(gs[0, 2:])
    ax_text.text(0.5, 0.5, 
                 "JEPA Depth Pipeline\n\n"
                 "1. Load paired RGB + Depth\n"
                 "2. Generate synchronized random crops\n"
                 "   • Global views (224×224): Large context\n"
                 "   • Local views (96×96): Fine details\n"
                 "3. Same crop applied to both RGB & Depth\n"
                 "4. ViT encodes each view → embeddings\n"
                 "5. LeJEPA: all views → match global center\n"
                 "6. Decoder: embeddings → depth prediction",
                 ha='center', va='center', fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                 transform=ax_text.transAxes)
    ax_text.axis('off')
    
    # Row 1: Global view RGB crops
    for i in range(V_global):
        ax = fig.add_subplot(gs[1, i])
        img_vis = denormalize_image(img_views[i])
        ax.imshow(img_vis)
        ax.set_title(f"Global View {i+1}\n{img_views[i].shape[-1]}×{img_views[i].shape[-2]}", fontsize=9)
        ax.axis('off')
        if i == 0:
            ax.set_ylabel("RGB", fontsize=10, rotation=0, ha='right', va='center')
    
    # Row 1: Local view RGB crops
    for i in range(V_local):
        ax = fig.add_subplot(gs[1, V_global + i])
        img_vis = denormalize_image(img_views[V_global + i])
        ax.imshow(img_vis)
        ax.set_title(f"Local View {i+1}\n{img_views[V_global + i].shape[-1]}×{img_views[V_global + i].shape[-2]}", fontsize=9)
        ax.axis('off')
    
    # Row 2: Global depth crops
    for i in range(V_global):
        ax = fig.add_subplot(gs[2, i])
        depth_vis = depth_to_colormap(depth_views[i])
        ax.imshow(depth_vis)
        ax.set_title(f"Depth (Global {i+1})", fontsize=9)
        ax.axis('off')
        if i == 0:
            ax.set_ylabel("Depth", fontsize=10, rotation=0, ha='right', va='center')
    
    # Row 2: Local depth crops
    for i in range(V_local):
        ax = fig.add_subplot(gs[2, V_global + i])
        depth_vis = depth_to_colormap(depth_views[V_global + i])
        ax.imshow(depth_vis)
        ax.set_title(f"Depth (Local {i+1})", fontsize=9)
        ax.axis('off')
    
    # Row 3: Explanation of JEPA loss
    ax_loss = fig.add_subplot(gs[3, :])
    ax_loss.text(0.5, 0.5,
                 "LeJEPA Loss Computation:\n\n"
                 "[1] Encode all views -> embeddings: [z1, z2, ..., z6]\n"
                 "[2] Center = mean(global embeddings) = mean([z1, z2])\n"
                 "[3] Prediction Loss = MSE(center, all views) -- all views should predict the same thing\n"
                 "[4] SIGReg Loss = Gaussian regularization on each embedding -- prevents collapse\n"
                 "[5] Depth Loss = Scale-Invariant Loss on predicted vs ground truth depth\n\n"
                 "Total = depth_weight * Depth Loss + jepa_weight * ((1-lambda) * Prediction Loss + lambda * SIGReg Loss)",
                 ha='center', va='center', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                 family='monospace',
                 transform=ax_loss.transAxes)
    ax_loss.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved pipeline visualization to {save_path}")


def visualize_with_predictions(dataset, model, device, idx=0, save_path="pipeline_predictions.png"):
    """
    Visualize predictions from a trained model.
    """
    from depth_model import DepthViT
    
    # Get sample
    img_views, depth_views = dataset[idx]
    
    V_global = dataset.V_global
    V_local = dataset.V_local
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        predictions = []
        for img in img_views:
            img_batch = img.unsqueeze(0).to(device, dtype=torch.bfloat16)
            pred = model(img_batch, return_embedding=False)
            predictions.append(pred.squeeze(0).cpu().float())
    
    # Create figure
    n_views = V_global + V_local
    fig, axes = plt.subplots(3, n_views, figsize=(3 * n_views, 9))
    
    for i in range(n_views):
        # Row 0: RGB
        img_vis = denormalize_image(img_views[i])
        axes[0, i].imshow(img_vis)
        view_type = "Global" if i < V_global else "Local"
        view_num = i + 1 if i < V_global else i - V_global + 1
        axes[0, i].set_title(f"{view_type} {view_num}\n{img_views[i].shape[-1]}px", fontsize=9)
        axes[0, i].axis('off')
        
        # Row 1: Ground Truth Depth
        gt_vis = depth_to_colormap(depth_views[i])
        axes[1, i].imshow(gt_vis)
        axes[1, i].set_title("GT Depth", fontsize=9)
        axes[1, i].axis('off')
        
        # Row 2: Predicted Depth
        # Resize prediction to match target size
        pred = predictions[i]
        if pred.shape[-1] != depth_views[i].shape[-1]:
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(0), 
                size=depth_views[i].shape[-2:],
                mode='bilinear'
            ).squeeze(0)
        pred_vis = depth_to_colormap(pred)
        axes[2, i].imshow(pred_vis)
        axes[2, i].set_title("Predicted", fontsize=9)
        axes[2, i].axis('off')
    
    axes[0, 0].set_ylabel("RGB Input", fontsize=10)
    axes[1, 0].set_ylabel("Ground Truth", fontsize=10)
    axes[2, 0].set_ylabel("Prediction", fontsize=10)
    
    plt.suptitle("JEPA Depth Estimation: Multi-View Predictions", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved prediction visualization to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize JEPA Depth Pipeline")
    parser.add_argument("--checkpoint", type=str, default=None, 
                       help="Path to trained model checkpoint")
    parser.add_argument("--output", type=str, default="pipeline_vis.png",
                       help="Output image path")
    parser.add_argument("--sample_idx", type=int, default=0,
                       help="Dataset sample index to visualize")
    parser.add_argument("--neighborhoods", type=str, default="0,1,2",
                       help="Comma-separated neighborhood IDs")
    args = parser.parse_args()
    
    # Parse neighborhoods
    neighborhoods = [int(n) for n in args.neighborhoods.split(",")]
    
    # Create dataset
    print("Loading dataset...")
    dataset = DepthDataset(
        split="train",
        global_img_size=224,
        local_img_size=96,
        neighborhoods=neighborhoods,
        V_global=2,
        V_local=4,
        multi_view=True,
    )
    
    if len(dataset) == 0:
        print("Error: No samples found in dataset!")
        return
    
    print(f"Dataset has {len(dataset)} samples")
    
    # Visualize dataset sample
    visualize_dataset_sample(dataset, idx=args.sample_idx, save_path=args.output)
    
    # If checkpoint provided, also show predictions
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading model from {args.checkpoint}...")
        from depth_model import DepthViT
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model = DepthViT(
            model_name="vit_small_patch16_224.augreg_in21k",
            img_size=224,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(torch.bfloat16)
        
        pred_output = args.output.replace(".png", "_predictions.png")
        visualize_with_predictions(dataset, model, device, idx=args.sample_idx, save_path=pred_output)


if __name__ == "__main__":
    main()
