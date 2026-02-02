
import argparse
import os
import sys
# Add project root to path
sys.path.append(os.getcwd())

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import wandb

from src.data.depth_ds import DepthDataset, collate_depth
from src.models.depth_model import DepthViT
from src.utils.metrics import compute_metrics
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--img_size", type=int, default=224, help="Patch size")
    parser.add_argument("--local_img_size", type=int, default=96)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="test_results")
    parser.add_argument("--model_name", type=str, default="vit_small_patch16_224.augreg_in21k")
    return parser.parse_args()

def reconstruct_from_patches(patches, original_shape, patch_size):
    """
    Reconstruct image from patches.
    patches: (L, C, P, P)
    original_shape: (H_orig, W_orig) tensor or tuple of pre-padding size
    patch_size: int (P)
    
    Returns: (C, H_orig, W_orig)
    """
    # patches is (L, C, P, P)
    L, C, P, P_ = patches.shape
    assert P == patch_size
    
    # Calculate padded dimensions
    H, W = original_shape
    if isinstance(H, torch.Tensor): H, W = H.item(), W.item()
    
    pad_h = (P - H % P) % P
    pad_w = (P - W % P) % P
    H_pad = H + pad_h
    W_pad = W + pad_w
    
    # Verify L matches
    n_h = H_pad // P
    n_w = W_pad // P
    if L != n_h * n_w:
        print(f"Warning: Patch count mismatch. Expected {n_h*n_w} ({n_h}x{n_w}), got {L}")
        # Could be due to last batch drop or something, but here patches are from single image
        # If mismatch, we might crash on fold.
        # But DepthDataset padding logic should guarantee match.

    # Prepare for fold
    # patches flat: (L, C*P*P)
    patches_flat = patches.view(L, -1).t().unsqueeze(0) # (1, C*P*P, L)
    
    # Fold
    # We used unfold with kernel=P, stride=P. So fold with same.
    recon = F.fold(
        patches_flat,
        output_size=(H_pad, W_pad),
        kernel_size=P,
        stride=P
    )
    # recon is (1, C, H_pad, W_pad)
    
    # Crop to original size
    recon = recon.squeeze(0)[:, :H, :W]
    return recon

def visualize(img, depth_gt, depth_pred, naive_pred, save_path):
    # img: (3, H, W) normalized
    # depth: (1, H, W) normalized [0,1]
    
    # Denormalize Image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img.cpu() * std + mean
    img = img.clamp(0, 1)
    
    depth_gt = depth_gt.cpu().squeeze(0)
    depth_pred = depth_pred.cpu().squeeze(0)
    naive_pred = naive_pred.cpu().squeeze(0)
    
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(img.permute(1, 2, 0))
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(depth_gt, cmap='inferno')
    plt.title("Ground Truth Depth")
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(naive_pred, cmap='inferno')
    plt.title("Naive Prediction (Gradient)")
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(depth_pred, cmap='inferno')
    plt.title("Model Prediction")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load Model
    # Load Model
    print(f"Loading model from {args.model_path}")
    
    checkpoint = torch.load(args.model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
        
    # Remove _orig_mod prefix if present (from torch.compile)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k.replace("_orig_mod.", "")] = v
        else:
            new_state_dict[k] = v
            
    print("Detected 5-layer decoder in checkpoint. Re-instantiating model with num_upsample=5...")
    model = DepthViT(
        model_name=args.model_name, 
        img_size=args.img_size,
        pretrained=True,
        num_upsample=5
    ).to(device)
    model.load_state_dict(new_state_dict, strict=True)
    
    
    model.to(torch.bfloat16) # Match training dtype
    model.eval()
    
    print("Loading dataset...")
    # split="test" uses Deterministic patching
    ds = DepthDataset(
        split="test", 
        global_img_size=args.img_size, 
        multi_view=False
    )
    
    loader = DataLoader(
        ds, 
        batch_size=8, # 8 images per batch (Total patches will be large)
        collate_fn=collate_depth, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    metrics_acc = {"abs_rel": 0, "sq_diff": 0, "n_pixels": 0}
    naive_acc = {"abs_rel": 0, "sq_diff": 0, "n_pixels": 0}
    
    visualized_count = 0
    
    print("Starting inference...")
    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc="Test"):
            # images: (Total_Patches, 3, P, P)
            p_images, p_depths, patch_counts, shapes = batch
            
            p_images = p_images.to(device, dtype=torch.bfloat16)
            p_depths = p_depths.to(device, dtype=torch.bfloat16)
            
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                p_preds = model(p_images, return_embedding=False)
            
            # Process image by image
            curr_idx = 0
            for i, count in enumerate(patch_counts):
                H_orig, W_orig = shapes[i]
                
                # Extract patches for this image
                img_patches = p_images[curr_idx : curr_idx+count]
                depth_patches = p_depths[curr_idx : curr_idx+count]
                pred_patches = p_preds[curr_idx : curr_idx+count]
                curr_idx += count
                
                # Reconstruct
                # reconstruct requires float32 mainly for fold, but bfloat16 works too usually
                img_recon = reconstruct_from_patches(img_patches, (H_orig, W_orig), args.img_size).float()
                depth_recon = reconstruct_from_patches(depth_patches, (H_orig, W_orig), args.img_size).float()
                pred_recon = reconstruct_from_patches(pred_patches, (H_orig, W_orig), args.img_size).float()
                
                # Naive prediction: Gradient 0->1
                # Note: Depth is normalized [0,1].
                y = torch.linspace(1,0, int(H_orig)).view(-1, 1).repeat(1, int(W_orig)).to(device)
                naive_pred = y.unsqueeze(0) # (1, H, W)
                
                # Metrics
                m = compute_metrics(pred_recon.unsqueeze(0), depth_recon.unsqueeze(0))
                nm = compute_metrics(naive_pred.unsqueeze(0), depth_recon.unsqueeze(0))
                
                metrics_acc["abs_rel"] += m["abs_rel_sum"]
                metrics_acc["sq_diff"] += m["sq_diff_sum"]
                metrics_acc["n_pixels"] += m["n_pixels"]
                
                naive_acc["abs_rel"] += nm["abs_rel_sum"]
                naive_acc["sq_diff"] += nm["sq_diff_sum"]
                naive_acc["n_pixels"] += nm["n_pixels"]
                
                # Visualize 3 samples
                if visualized_count < 3:
                     visualize(img_recon, depth_recon, pred_recon, naive_pred, 
                               os.path.join(args.save_dir, f"result_{visualized_count}.png"))
                     visualized_count += 1
                     
    # Final Metrics
    print("\n=== Test Results ===")
    if metrics_acc["n_pixels"] > 0:
        model_abs_rel = metrics_acc['abs_rel'] / metrics_acc['n_pixels']
        model_rmse = (metrics_acc['sq_diff'] / metrics_acc['n_pixels'])**0.5
        print(f"Model AbsRel: {model_abs_rel:.4f}")
        print(f"Model RMSE:   {model_rmse:.4f}")
    
    if naive_acc["n_pixels"] > 0:
        naive_abs_rel = naive_acc['abs_rel'] / naive_acc['n_pixels']
        naive_rmse = (naive_acc['sq_diff'] / naive_acc['n_pixels'])**0.5
        print(f"Naive AbsRel: {naive_abs_rel:.4f}")
        print(f"Naive RMSE:   {naive_rmse:.4f}")

if __name__ == "__main__":
    main()
