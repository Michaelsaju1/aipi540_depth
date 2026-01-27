"""
Training script for Depth Estimation with ViT + SIGReg regularization.

Usage:
    python train_depth.py --epochs 50 --bs 16 --lr 1e-4

For RTX Pro 6000 (48GB), recommended settings:
    - vit_small: bs=32, img_size=224
    - vit_base:  bs=16, img_size=224
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import argparse
import logging
import tqdm
import wandb

from depth_ds import DepthDataset, collate_depth
from depth_model import DepthViT, ScaleInvariantLoss, DepthSmoothL1Loss
from loss import SIGReg

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Train depth estimation model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--bs", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--model", type=str, default="vit_small_patch16_224.augreg_in21k")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--sigreg_weight", type=float, default=0.1, help="Weight for SIGReg loss")
    parser.add_argument("--neighborhoods", type=str, default=None, 
                       help="Comma-separated list of neighborhood IDs, e.g. '0,1,2'")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--save_path", type=str, default="checkpoints/depth_model.pt")
    return parser.parse_args()


def compute_metrics(pred, target):
    """Compute depth estimation metrics."""
    # Ensure same shape
    pred = pred.detach()
    target = target.detach()
    
    # Avoid division by zero
    eps = 1e-6
    pred = pred.clamp(min=eps)
    target = target.clamp(min=eps)
    
    # Absolute Relative Error
    abs_rel = torch.mean(torch.abs(pred - target) / target)
    
    # RMSE
    rmse = torch.sqrt(torch.mean((pred - target) ** 2))
    
    # Delta thresholds (% of pixels where max(pred/gt, gt/pred) < threshold)
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
        img_size=args.img_size,
        neighborhoods=neighborhoods,
    )
    val_ds = DepthDataset(
        split="val",
        img_size=args.img_size,
        neighborhoods=neighborhoods,
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_depth,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_depth,
    )
    
    logging.info(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
    
    # Model
    logging.info(f"Creating model: {args.model}")
    model = DepthViT(
        model_name=args.model,
        img_size=args.img_size,
        pretrained=True,
    ).to(device)
    
    # Convert to bfloat16 for efficiency
    model = model.to(torch.bfloat16)
    
    # Compile for speed (PyTorch 2.0+)
    # model = torch.compile(model, mode="reduce-overhead")
    
    # SIGReg for embedding regularization
    sigreg = SIGReg().to(device)
    
    # Loss functions
    depth_loss_fn = ScaleInvariantLoss(lambd=0.5)
    # Alternative: DepthSmoothL1Loss(edge_weight=0.5)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.05,
        betas=(0.9, 0.95),
    )
    
    # Scheduler
    steps_per_epoch = len(train_loader) // args.grad_accum
    warmup_steps = steps_per_epoch  # 1 epoch warmup
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
            project="AIPI_540_Depth",
            name=f"depth_{args.model.split('.')[0]}_bs{args.bs}",
            config=vars(args),
        )
    
    # Training loop
    global_step = 0
    best_delta1 = 0
    scaler = GradScaler()
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        optimizer.zero_grad()
        
        for batch_idx, (images, depths) in enumerate(pbar):
            images = images.to(device, dtype=torch.bfloat16, non_blocking=True)
            depths = depths.to(device, dtype=torch.bfloat16, non_blocking=True)
            
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                # Forward
                pred_depth, embedding = model(images, return_embedding=True)
                
                # Depth loss
                depth_loss = depth_loss_fn(pred_depth, depths)
                
                # SIGReg on embeddings (encourages Gaussian distribution)
                sigreg_loss = sigreg(embedding.float())
                
                # Total loss
                loss = depth_loss + args.sigreg_weight * sigreg_loss
                loss = loss / args.grad_accum
            
            # Backward
            loss.backward()
            
            # Gradient accumulation step
            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            epoch_loss += depth_loss.item()
            
            # Logging
            if global_step % 20 == 0:
                log_dict = {
                    "train/depth_loss": depth_loss.item(),
                    "train/sigreg_loss": sigreg_loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
                if args.wandb:
                    wandb.log(log_dict, step=global_step)
                pbar.set_postfix(loss=depth_loss.item(), lr=optimizer.param_groups[0]["lr"])
            
            global_step += 1
        
        avg_loss = epoch_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}")
        
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
        
        # Average metrics
        for k in val_metrics:
            val_metrics[k] /= num_val
        
        logging.info(f"Val - AbsRel: {val_metrics['abs_rel']:.4f}, RMSE: {val_metrics['rmse']:.4f}, "
                    f"δ1: {val_metrics['delta1']:.4f}, δ2: {val_metrics['delta2']:.4f}")
        
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
            }, args.save_path)
            logging.info(f"Saved best model with δ1={best_delta1:.4f}")
    
    if args.wandb:
        wandb.finish()
    
    logging.info(f"Training complete. Best δ1: {best_delta1:.4f}")


if __name__ == "__main__":
    main()
