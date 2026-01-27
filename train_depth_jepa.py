"""
Training script for Depth Estimation with JEPA (LeJEPA) + SIGReg regularization.

This implements the same loss function logic as run_JEPA.py and loss.py:
- Multi-view learning: global views (224px) + local views (96px)
- LeJEPA loss: center prediction + SIGReg regularization
- Depth supervision on all views

Usage:
    python train_depth_jepa.py --epochs 50 --bs 16 --lr 1e-4

For RTX Pro 6000 (48GB), recommended settings:
    - vit_small: bs=16, V_global=2, V_local=4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import argparse
import logging
import tqdm
import wandb

from depth_ds import DepthDataset, collate_depth, collate_depth_multiview
from depth_model import DepthViT, ScaleInvariantLoss
from loss import SIGReg

logging.basicConfig(level=logging.INFO)


def LeJEPA_Depth(global_proj, all_proj, sigreg, lamb):
    """
    LeJEPA loss for depth estimation.
    
    Same logic as loss.py LeJEPA but adapted for our pipeline.
    
    Args:
        global_proj: (N, Vg, D) - Embeddings of global views
        all_proj: (N, V, D) - Embeddings of all views (global + local)
        sigreg: SIGReg module for regularization
        lamb: Weight for SIGReg loss (typically 0.05-0.1)
    
    Returns:
        total_loss: Combined (1-lamb)*sim_loss + lamb*sigreg_loss
        sim_loss: Prediction loss (MSE between centers and all views)
        sigreg_loss: SIGReg regularization loss
    """
    # Centers from global views
    centers = global_proj.mean(dim=1, keepdim=True)  # (N, 1, D)
    
    # Prediction loss: MSE between centers and all views
    # Encourages all views to predict the same representation
    sim_loss = (centers - all_proj).square().mean()
    
    # SIGReg loss on each view embedding
    # Encourages Gaussian distribution in projection space
    sigreg_losses = []
    for i in range(all_proj.shape[1]):
        view_emb = all_proj[:, i, :]  # (N, D)
        # SIGReg expects float32
        l = sigreg(view_emb.float())
        sigreg_losses.append(l)
    sigreg_loss = torch.stack(sigreg_losses).mean()
    
    total_loss = (1 - lamb) * sim_loss + lamb * sigreg_loss
    
    return total_loss, sim_loss, sigreg_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train depth estimation with JEPA")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--bs", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--global_img_size", type=int, default=224)
    parser.add_argument("--local_img_size", type=int, default=96)
    parser.add_argument("--model", type=str, default="vit_small_patch16_224.augreg_in21k")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=1)
    
    # JEPA parameters
    parser.add_argument("--V_global", type=int, default=2, help="Number of global views")
    parser.add_argument("--V_local", type=int, default=4, help="Number of local views")
    parser.add_argument("--lamb", type=float, default=0.05, help="SIGReg weight in LeJEPA")
    
    # Loss weights
    parser.add_argument("--depth_weight", type=float, default=1.0, help="Weight for depth loss")
    parser.add_argument("--jepa_weight", type=float, default=0.5, help="Weight for JEPA loss")
    
    parser.add_argument("--neighborhoods", type=str, default=None,
                       help="Comma-separated list of neighborhood IDs")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--save_path", type=str, default="checkpoints/depth_jepa.pt")
    return parser.parse_args()


def compute_metrics(pred, target):
    """Compute depth estimation metrics."""
    pred = pred.detach()
    target = target.detach()
    
    eps = 1e-6
    pred = pred.clamp(min=eps)
    target = target.clamp(min=eps)
    
    abs_rel = torch.mean(torch.abs(pred - target) / target)
    rmse = torch.sqrt(torch.mean((pred - target) ** 2))
    
    ratio = torch.max(pred / target, target / pred)
    delta1 = (ratio < 1.25).float().mean()
    delta2 = (ratio < 1.25 ** 2).float().mean()
    delta3 = (ratio < 1.25 ** 3).float().mean()
    
    return {
        "abs_rel": abs_rel.item(),
        "rmse": rmse.item(),
        "delta1": delta1.item(),
        "delta2": delta2.item(),
        "delta3": delta3.item(),
    }


def main():
    args = parse_args()
    
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42)
    
    device = torch.device(args.device)
    
    # Parse neighborhoods
    neighborhoods = None
    if args.neighborhoods:
        neighborhoods = [int(n) for n in args.neighborhoods.split(",")]
    
    # Datasets
    logging.info("Loading datasets...")
    train_ds = DepthDataset(
        split="train",
        global_img_size=args.global_img_size,
        local_img_size=args.local_img_size,
        neighborhoods=neighborhoods,
        V_global=args.V_global,
        V_local=args.V_local,
        multi_view=True,  # Enable JEPA multi-view
    )
    
    # Validation uses single view
    val_ds = DepthDataset(
        split="val",
        global_img_size=args.global_img_size,
        local_img_size=args.local_img_size,
        neighborhoods=neighborhoods,
        multi_view=False,
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_depth_multiview,  # Multi-view collate for training
        persistent_workers=args.num_workers > 0,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_depth,  # Single view collate for validation
    )
    
    logging.info(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
    logging.info(f"Views: {args.V_global} global + {args.V_local} local")
    
    # Model
    logging.info(f"Creating model: {args.model}")
    model = DepthViT(
        model_name=args.model,
        img_size=args.global_img_size,
        pretrained=True,
    ).to(device)
    
    # Convert to bfloat16
    model = model.to(torch.bfloat16)
    
    # SIGReg module (same as in run_JEPA.py)
    sigreg = SIGReg().to(device)
    
    # Loss functions
    depth_loss_fn = ScaleInvariantLoss(lambd=0.5)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.05,
        betas=(0.9, 0.95),
    )
    
    # Scheduler
    steps_per_epoch = len(train_loader) // args.grad_accum
    warmup_steps = steps_per_epoch
    total_steps = steps_per_epoch * args.epochs
    
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps),
            CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6),
        ],
        milestones=[warmup_steps],
    )
    
    # W&B
    if args.wandb:
        wandb.init(
            project="AIPI_540_Depth_JEPA",
            name=f"jepa_{args.model.split('.')[0]}_V{args.V_global}+{args.V_local}",
            config=vars(args),
        )
    
    # Training loop
    global_step = 0
    best_delta1 = 0
    
    for epoch in range(args.epochs):
        model.train()
        epoch_depth_loss = 0
        epoch_jepa_loss = 0
        
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        optimizer.zero_grad()
        
        for batch_idx, (img_views, depth_views) in enumerate(pbar):
            # img_views: List of (B, 3, H, W) - global views first, then local
            # depth_views: List of (B, 1, H, W) - matching depth maps
            
            # Separate global and local views by size
            global_imgs = [v.to(device, dtype=torch.bfloat16, non_blocking=True) 
                          for v in img_views if v.shape[-1] == args.global_img_size]
            local_imgs = [v.to(device, dtype=torch.bfloat16, non_blocking=True)
                         for v in img_views if v.shape[-1] == args.local_img_size]
            
            global_depths = [v.to(device, dtype=torch.bfloat16, non_blocking=True)
                            for v in depth_views if v.shape[-1] == args.global_img_size]
            local_depths = [v.to(device, dtype=torch.bfloat16, non_blocking=True)
                           for v in depth_views if v.shape[-1] == args.local_img_size]
            
            all_imgs = global_imgs + local_imgs
            all_depths = global_depths + local_depths
            
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                # Forward pass for all views
                all_pred_depths = []
                all_embeddings = []
                
                for img in all_imgs:
                    pred_depth, emb = model(img, return_embedding=True)
                    all_pred_depths.append(pred_depth)
                    all_embeddings.append(emb)
                
                # Stack embeddings: (B, V, D)
                all_emb = torch.stack(all_embeddings, dim=1)  # (B, V, D)
                global_emb = all_emb[:, :len(global_imgs), :]  # (B, Vg, D)
                
                # === JEPA Loss (same as LeJEPA in loss.py) ===
                jepa_loss, sim_loss, sigreg_loss = LeJEPA_Depth(
                    global_emb, all_emb, sigreg, args.lamb
                )
                
                # === Depth Loss (on all views) ===
                depth_losses = []
                for pred, target in zip(all_pred_depths, all_depths):
                    # Resize prediction to match target if needed (for local views)
                    if pred.shape[-1] != target.shape[-1]:
                        pred = F.interpolate(pred, size=target.shape[-2:], 
                                           mode='bilinear', align_corners=False)
                    depth_losses.append(depth_loss_fn(pred, target))
                depth_loss = torch.stack(depth_losses).mean()
                
                # === Total Loss ===
                loss = args.depth_weight * depth_loss + args.jepa_weight * jepa_loss
                loss = loss / args.grad_accum
            
            # Backward
            loss.backward()
            
            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            epoch_depth_loss += depth_loss.item()
            epoch_jepa_loss += jepa_loss.item()
            
            # Logging
            if global_step % 20 == 0:
                log_dict = {
                    "train/depth_loss": depth_loss.item(),
                    "train/jepa_loss": jepa_loss.item(),
                    "train/sim_loss": sim_loss.item(),
                    "train/sigreg_loss": sigreg_loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
                if args.wandb:
                    wandb.log(log_dict, step=global_step)
                pbar.set_postfix(
                    depth=depth_loss.item(),
                    jepa=jepa_loss.item(),
                    lr=optimizer.param_groups[0]["lr"]
                )
            
            global_step += 1
        
        avg_depth = epoch_depth_loss / len(train_loader)
        avg_jepa = epoch_jepa_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1} - Depth: {avg_depth:.4f}, JEPA: {avg_jepa:.4f}")
        
        # Validation
        model.eval()
        val_metrics = {"abs_rel": 0, "rmse": 0, "delta1": 0, "delta2": 0, "delta3": 0}
        num_val = 0
        
        with torch.inference_mode():
            for images, depths in tqdm.tqdm(val_loader, desc="Validation"):
                images = images.to(device, dtype=torch.bfloat16, non_blocking=True)
                depths = depths.to(device, dtype=torch.bfloat16, non_blocking=True)
                
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred_depth = model(images, return_embedding=False)
                
                metrics = compute_metrics(pred_depth, depths)
                for k, v in metrics.items():
                    val_metrics[k] += v * images.size(0)
                num_val += images.size(0)
        
        for k in val_metrics:
            val_metrics[k] /= num_val
        
        logging.info(f"Val - AbsRel: {val_metrics['abs_rel']:.4f}, RMSE: {val_metrics['rmse']:.4f}, "
                    f"δ1: {val_metrics['delta1']:.4f}")
        
        if args.wandb:
            wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=global_step)
        
        # Save best model
        if val_metrics["delta1"] > best_delta1:
            best_delta1 = val_metrics["delta1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "args": vars(args),
            }, args.save_path)
            logging.info(f"Saved best model with δ1={best_delta1:.4f}")
    
    if args.wandb:
        wandb.finish()
    
    logging.info(f"Training complete. Best δ1: {best_delta1:.4f}")


if __name__ == "__main__":
    main()
