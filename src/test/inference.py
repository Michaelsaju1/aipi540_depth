"""
Inference script for depth segmentation on arbitrary-sized images.

This script:
1. Loads a trained ViT depth model
2. Accepts images of any size
3. Chunks them into model-compatible patches (e.g., 224x224)
4. Runs inference on each patch
5. Stitches results back together for the full depth map
"""

import argparse
import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn.functional as F
from torch.amp import autocast
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

from src.models.depth_model import DepthViT


def parse_args():
    parser = argparse.ArgumentParser(description="Run depth segmentation on arbitrary-sized images")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image or directory")
    parser.add_argument("--output_dir", type=str, default="inference_results", help="Output directory")
    parser.add_argument("--patch_size", type=int, default=224, help="Patch size for chunking (model input size)")
    parser.add_argument("--overlap", type=int, default=32, help="Overlap between patches for smoother blending")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on")
    parser.add_argument("--model_name", type=str, default="vit_small_patch16_224.augreg_in21k")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing patches")
    parser.add_argument("--colormap", type=str, default="inferno", help="Colormap for depth visualization")
    parser.add_argument("--save_raw", action="store_true", help="Save raw depth map as .npy")
    return parser.parse_args()


def load_model(model_path, model_name, device):
    """Load the trained ViT depth model."""
    print(f"Loading model from {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
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
    
    # Count decoder layers to determine num_upsample
    decoder_layers = [k for k in new_state_dict.keys() if k.startswith("decoder.net.")]
    num_upsample = len(set(k.split('.')[2] for k in decoder_layers))
    if num_upsample == 0:
        num_upsample = 5  # Default fallback
    print(f"Detected {num_upsample}-layer decoder in checkpoint")
    
    model = DepthViT(
        model_name=model_name,
        img_size=224,  # Standard ViT patch size
        pretrained=True,
        num_upsample=num_upsample
    ).to(device)
    
    model.load_state_dict(new_state_dict, strict=True)
    model.to(torch.bfloat16)
    model.eval()
    
    return model


def chunk_image(image_tensor, patch_size, overlap):
    """
    Chunk an image tensor into overlapping patches.
    
    Args:
        image_tensor: (C, H, W) tensor
        patch_size: Size of each patch
        overlap: Overlap between adjacent patches
        
    Returns:
        patches: List of (C, patch_size, patch_size) tensors
        positions: List of (y, x) positions for each patch
        padded_shape: (H_padded, W_padded) shape after padding
    """
    C, H, W = image_tensor.shape
    stride = patch_size - overlap
    
    # Calculate padding needed
    pad_h = (patch_size - (H - patch_size) % stride) % stride if H > patch_size else patch_size - H
    pad_w = (patch_size - (W - patch_size) % stride) % stride if W > patch_size else patch_size - W
    
    # For small images, ensure at least one patch
    if H <= patch_size:
        pad_h = patch_size - H
    if W <= patch_size:
        pad_w = patch_size - W
    
    # Pad image (right and bottom)
    padded = F.pad(image_tensor.unsqueeze(0), (0, pad_w, 0, pad_h), mode='reflect').squeeze(0)
    _, H_pad, W_pad = padded.shape
    
    patches = []
    positions = []
    
    y = 0
    while y + patch_size <= H_pad:
        x = 0
        while x + patch_size <= W_pad:
            patch = padded[:, y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            positions.append((y, x))
            
            # Move to next position
            if x + patch_size >= W_pad:
                break
            x += stride
            # Adjust last column to align with edge
            if x + patch_size > W_pad:
                x = W_pad - patch_size
                
        if y + patch_size >= H_pad:
            break
        y += stride
        # Adjust last row to align with edge
        if y + patch_size > H_pad:
            y = H_pad - patch_size
    
    return patches, positions, (H_pad, W_pad)


def stitch_patches(patches, positions, output_shape, patch_size, overlap):
    """
    Stitch depth patches back together with weighted blending.
    
    Args:
        patches: List of (1, patch_size, patch_size) depth tensors
        positions: List of (y, x) positions
        output_shape: (H, W) target output shape
        patch_size: Size of each patch
        overlap: Overlap between patches
        
    Returns:
        depth_map: (1, H, W) stitched depth map
    """
    H, W = output_shape
    device = patches[0].device
    dtype = patches[0].dtype
    
    # Create output tensor and weight accumulator
    depth_sum = torch.zeros(1, H, W, device=device, dtype=torch.float32)
    weight_sum = torch.zeros(1, H, W, device=device, dtype=torch.float32)
    
    # Create blending weights (linear ramp for overlap regions)
    weights = create_blend_weights(patch_size, overlap, device)
    
    for patch, (y, x) in zip(patches, positions):
        patch_float = patch.float()
        
        # Handle edge cases where patch extends beyond output
        y_end = min(y + patch_size, H)
        x_end = min(x + patch_size, W)
        patch_h = y_end - y
        patch_w = x_end - x
        
        depth_sum[:, y:y_end, x:x_end] += patch_float[:, :patch_h, :patch_w] * weights[:, :patch_h, :patch_w]
        weight_sum[:, y:y_end, x:x_end] += weights[:, :patch_h, :patch_w]
    
    # Avoid division by zero
    weight_sum = torch.clamp(weight_sum, min=1e-6)
    depth_map = depth_sum / weight_sum
    
    return depth_map


def create_blend_weights(patch_size, overlap, device):
    """
    Create blending weights for smooth patch stitching.
    Uses linear ramps in overlap regions.
    """
    if overlap == 0:
        return torch.ones(1, patch_size, patch_size, device=device)
    
    # Create 1D ramp
    ramp = torch.ones(patch_size, device=device)
    ramp[:overlap] = torch.linspace(0, 1, overlap, device=device)
    ramp[-overlap:] = torch.linspace(1, 0, overlap, device=device)
    
    # Create 2D weights
    weights = ramp.unsqueeze(0) * ramp.unsqueeze(1)
    return weights.unsqueeze(0)


def preprocess_image(image_path):
    """Load and preprocess an image."""
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (W, H)
    
    # Standard ImageNet normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tensor = transform(image)
    return tensor, original_size


def postprocess_depth(depth_tensor, original_size):
    """Crop and resize depth map back to original image size."""
    # depth_tensor: (1, H_padded, W_padded)
    # original_size: (W, H)
    W_orig, H_orig = original_size
    
    # Crop to remove padding
    depth_cropped = depth_tensor[:, :H_orig, :W_orig]
    
    return depth_cropped


def visualize_depth(image_path, depth_map, save_path, colormap='inferno'):
    """Create visualization comparing input image and predicted depth."""
    # Load original image
    image = Image.open(image_path).convert("RGB")
    
    depth_np = depth_map.squeeze().cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis('off')
    
    im = axes[1].imshow(depth_np, cmap=colormap)
    axes[1].set_title("Predicted Depth")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {save_path}")


def run_inference(model, image_tensor, args, device):
    """
    Run inference on an arbitrary-sized image.
    
    Args:
        model: Trained ViT depth model
        image_tensor: (C, H, W) preprocessed image tensor
        args: Command line arguments
        device: Torch device
        
    Returns:
        depth_map: (1, H, W) predicted depth map
    """
    C, H, W = image_tensor.shape
    
    # Chunk the image into patches
    patches, positions, padded_shape = chunk_image(
        image_tensor, args.patch_size, args.overlap
    )
    
    print(f"Image size: {H}x{W}, Padded: {padded_shape[0]}x{padded_shape[1]}")
    print(f"Created {len(patches)} patches with {args.overlap}px overlap")
    
    # Stack patches for batched inference
    patches_tensor = torch.stack(patches).to(device, dtype=torch.bfloat16)
    
    # Run inference in batches
    depth_patches = []
    with torch.no_grad():
        for i in range(0, len(patches_tensor), args.batch_size):
            batch = patches_tensor[i:i + args.batch_size]
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                batch_depth = model(batch, return_embedding=False)
            depth_patches.extend([d for d in batch_depth])
    
    # Stitch patches back together
    depth_map = stitch_patches(
        depth_patches, positions, padded_shape, args.patch_size, args.overlap
    )
    
    return depth_map


def process_single_image(model, image_path, args, device):
    """Process a single image and save results."""
    print(f"\nProcessing: {image_path}")
    
    # Preprocess
    image_tensor, original_size = preprocess_image(image_path)
    
    # Run inference
    depth_map = run_inference(model, image_tensor, args, device)
    
    # Postprocess (crop to original size)
    depth_map = postprocess_depth(depth_map, original_size)
    
    # Create output filename
    basename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save visualization
    vis_path = os.path.join(args.output_dir, f"{basename}_depth.png")
    visualize_depth(image_path, depth_map, vis_path, args.colormap)
    
    # Optionally save raw depth
    if args.save_raw:
        raw_path = os.path.join(args.output_dir, f"{basename}_depth.npy")
        np.save(raw_path, depth_map.squeeze().cpu().numpy())
        print(f"Saved raw depth to {raw_path}")
    
    return depth_map


def main():
    args = parse_args()
    device = torch.device(args.device)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args.model_path, args.model_name, device)
    
    # Check if input is file or directory
    if os.path.isfile(args.image_path):
        # Single image
        process_single_image(model, args.image_path, args, device)
    elif os.path.isdir(args.image_path):
        # Directory of images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [
            f for f in os.listdir(args.image_path)
            if os.path.splitext(f)[1].lower() in image_extensions
        ]
        
        print(f"Found {len(image_files)} images in {args.image_path}")
        
        for image_file in sorted(image_files):
            image_path = os.path.join(args.image_path, image_file)
            try:
                process_single_image(model, image_path, args, device)
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                continue
    else:
        print(f"Error: {args.image_path} is not a valid file or directory")
        sys.exit(1)
    
    print(f"\nDone! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
