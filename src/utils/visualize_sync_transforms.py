#!/usr/bin/env python3
"""
Visualization script to verify synchronized transforms.

This script loads samples from the MultiTaskDataset and visualizes them
to ensure that image, depth, and segmentation are properly aligned after
augmentation.

Usage:
    python src/utils/visualize_sync_transforms.py --dataset neighbourhood --scene_id 0 --num_samples 5
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

from src.data.ds import MultiTaskDataset


def denormalize_image(img_tensor):
    """Denormalize image from ImageNet normalization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor * std + mean
    return torch.clamp(img, 0, 1)


def visualize_sample(sample, idx, save_path=None):
    """
    Visualize a single sample with image, depth, and segmentation.
    
    Args:
        sample: Dict with 'image', 'depth', 'segmentation' keys
        idx: Sample index for title
        save_path: Optional path to save the figure
    """
    # Denormalize image
    image = denormalize_image(sample['image'])
    depth = sample['depth']
    segmentation = sample['segmentation']
    
    # Convert to numpy for visualization
    image_np = image.permute(1, 2, 0).numpy()
    depth_np = depth.squeeze(0).numpy()
    seg_np = segmentation.squeeze(0).numpy()
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Plot image
    axes[0].imshow(image_np)
    axes[0].set_title(f'Sample {idx}: RGB Image')
    axes[0].axis('off')
    
    # Plot depth
    im1 = axes[1].imshow(depth_np, cmap='plasma')
    axes[1].set_title('Depth Map')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    # Plot segmentation
    im2 = axes[2].imshow(seg_np, cmap='tab20')
    axes[2].set_title('Segmentation')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    
    # Plot overlay: image with segmentation boundaries
    axes[3].imshow(image_np)
    axes[3].imshow(seg_np, cmap='tab20', alpha=0.3)
    axes[3].set_title('Image + Segmentation Overlay')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def verify_alignment(sample):
    """
    Verify that transforms are synchronized by checking alignment.
    
    For augmented samples, we can check if the flip was applied consistently
    by looking at the spatial correspondence between modalities.
    """
    image = sample['image']
    depth = sample['depth']
    seg = sample['segmentation']
    
    print(f"Image shape: {image.shape}")
    print(f"Depth shape: {depth.shape}")
    print(f"Segmentation shape: {seg.shape}")
    print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"Depth range: [{depth.min():.3f}, {depth.max():.3f}]")
    print(f"Segmentation unique values: {seg.unique().tolist()[:10]}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Visualize synchronized transforms")
    parser.add_argument('--dataset', type=str, default='neighbourhood', 
                       choices=['neighbourhood', 'ddos'])
    parser.add_argument('--scene_id', type=int, default=0, 
                       help='Scene ID for neighbourhood dataset')
    parser.add_argument('--num_samples', type=int, default=5, 
                       help='Number of samples to visualize')
    parser.add_argument('--augment', action='store_true', 
                       help='Show augmented samples (train mode)')
    parser.add_argument('--output_dir', type=str, default='outputs/sync_transforms',
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dataset
    split = 'train' if args.augment else 'test'
    print(f"Loading {args.dataset} dataset (split={split}, augment={args.augment})...")
    
    dataset = MultiTaskDataset(
        split=split,
        dataset=args.dataset,
        scene_id=args.scene_id if args.dataset == 'neighbourhood' else None,
        img_size=224,
        augment=args.augment
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Visualizing {args.num_samples} samples...\n")
    
    # Visualize samples
    for i in range(min(args.num_samples, len(dataset))):
        print(f"--- Sample {i} ---")
        sample = dataset[i]
        
        # Verify alignment
        verify_alignment(sample)
        
        # Visualize
        save_path = os.path.join(
            args.output_dir, 
            f"{args.dataset}_sample_{i}_{'aug' if args.augment else 'test'}.png"
        )
        visualize_sample(sample, i, save_path)
    
    print(f"\nAll visualizations saved to {args.output_dir}")
    
    # Test with multiple augmented versions of the same sample
    if args.augment:
        print("\n=== Testing Multiple Augmentations ===")
        print("Loading the same sample 3 times with different augmentations...")
        
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        
        for aug_idx in range(3):
            sample = dataset[0]  # Same sample, different augmentation
            
            image = denormalize_image(sample['image']).permute(1, 2, 0).numpy()
            depth = sample['depth'].squeeze(0).numpy()
            seg = sample['segmentation'].squeeze(0).numpy()
            
            axes[aug_idx, 0].imshow(image)
            axes[aug_idx, 0].set_title(f'Aug {aug_idx+1}: Image')
            axes[aug_idx, 0].axis('off')
            
            axes[aug_idx, 1].imshow(depth, cmap='plasma')
            axes[aug_idx, 1].set_title('Depth')
            axes[aug_idx, 1].axis('off')
            
            axes[aug_idx, 2].imshow(seg, cmap='tab20')
            axes[aug_idx, 2].set_title('Segmentation')
            axes[aug_idx, 2].axis('off')
            
            axes[aug_idx, 3].imshow(image)
            axes[aug_idx, 3].imshow(seg, cmap='tab20', alpha=0.3)
            axes[aug_idx, 3].set_title('Overlay')
            axes[aug_idx, 3].axis('off')
        
        plt.suptitle('Same Sample with Different Augmentations', fontsize=16)
        plt.tight_layout()
        
        aug_path = os.path.join(args.output_dir, 'multiple_augmentations.png')
        plt.savefig(aug_path, dpi=150, bbox_inches='tight')
        print(f"Saved multiple augmentations to {aug_path}")
        plt.close()


if __name__ == "__main__":
    main()
