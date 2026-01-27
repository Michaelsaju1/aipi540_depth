"""
Depth Estimation Dataset for DDOS neighborhood data with JEPA-style multi-view support.

Loads RGB images and corresponding depth maps with synchronized transforms
to ensure the same random crop is applied to both.

Supports:
- Single view mode (for simple depth prediction)
- Multi-view mode (for JEPA training with global + local views)
"""

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import os
import glob
from PIL import Image
import random
import numpy as np


class DepthDataset(Dataset):
    """
    Dataset for depth estimation from RGB images.
    
    Loads paired (RGB, Depth) from neighbourhood folders.
    Applies identical random crops to both image and depth map.
    
    For JEPA training, generates multiple views (global + local) with
    synchronized transforms applied to both RGB and depth.
    """
    
    def __init__(
        self, 
        split="train",
        global_img_size=224,
        local_img_size=96,
        data_root="datalink/neighbourhood",
        neighborhoods=None,  # List of neighborhood IDs, e.g. [0, 1, 2] or None for all
        depth_max=65535.0,   # Max depth value for normalization
        V_global=2,          # Number of global views (for JEPA)
        V_local=4,           # Number of local views (for JEPA)
        multi_view=True,     # Enable multi-view for JEPA training
    ):
        self.split = split
        self.global_img_size = global_img_size
        self.local_img_size = local_img_size
        self.depth_max = depth_max
        self.data_root = data_root
        self.V_global = V_global
        self.V_local = V_local
        self.multi_view = multi_view
        
        # Collect all image/depth pairs
        self.samples = self._collect_samples(neighborhoods)
        
        # Split train/val (80/20)
        rng = random.Random(42)
        indices = list(range(len(self.samples)))
        rng.shuffle(indices)
        split_idx = int(0.8 * len(indices))
        
        if split == "train":
            self.samples = [self.samples[i] for i in indices[:split_idx]]
        else:
            self.samples = [self.samples[i] for i in indices[split_idx:]]
        
        print(f"DepthDataset [{split}]: {len(self.samples)} samples, "
              f"multi_view={multi_view}, V_global={V_global}, V_local={V_local}")
        
        # Color augmentation (applied only to RGB, NOT depth)
        self.color_aug = v2.Compose([
            v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            v2.RandomGrayscale(p=0.1),
        ])
        
    def _resolve_symlink(self, symlink_path):
        """
        Resolve a symlink to its actual blob path.
        
        The neighbourhood symlinks point to ../../../../../../../blobs/HASH
        but the blobs are actually in datalink/cache/datasets--benediktkol--DDOS/blobs/
        """
        if not os.path.islink(symlink_path):
            return symlink_path if os.path.exists(symlink_path) else None
        
        target = os.readlink(symlink_path)
        # Extract the hash from the symlink target
        blob_hash = os.path.basename(target)
        
        # Construct the correct blob path
        blob_path = os.path.join(
            os.path.dirname(self.data_root),  # datalink
            "cache", "datasets--benediktkol--DDOS", "blobs", blob_hash
        )
        
        if os.path.exists(blob_path):
            return blob_path
        return None
    
    def _collect_samples(self, neighborhoods):
        """Gather all (image_path, depth_path) pairs."""
        samples = []
        
        # Get all neighborhood folders
        if neighborhoods is None:
            folders = [d for d in os.listdir(self.data_root) 
                      if os.path.isdir(os.path.join(self.data_root, d)) and d.isdigit()]
        else:
            folders = [str(n) for n in neighborhoods]
        
        for folder in folders:
            img_dir = os.path.join(self.data_root, folder, "image")
            depth_dir = os.path.join(self.data_root, folder, "depth")
            
            if not os.path.isdir(img_dir) or not os.path.isdir(depth_dir):
                continue
            
            # Get all image files (symlinks)
            for img_name in os.listdir(img_dir):
                if not img_name.endswith('.png'):
                    continue
                    
                img_symlink = os.path.join(img_dir, img_name)
                depth_symlink = os.path.join(depth_dir, img_name)
                
                # Resolve symlinks to actual blob paths
                img_file = self._resolve_symlink(img_symlink)
                depth_file = self._resolve_symlink(depth_symlink)
                
                if img_file and depth_file:
                    samples.append((img_file, depth_file))
        
        return samples
    
    def _synchronized_crop(self, img, depth, output_size, scale=(0.5, 1.0), ratio=(0.75, 1.33)):
        """
        Apply identical random resized crop to both image and depth.
        
        Args:
            img: PIL Image (RGB)
            depth: PIL Image (depth map)
            output_size: Target size (H, W) or int
            scale: Scale range for random resized crop
            ratio: Aspect ratio range
        
        Returns:
            img_crop: PIL Image
            depth_crop: PIL Image
        """
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
            
        # Get random crop parameters (shared between img and depth)
        i, j, h, w = v2.RandomResizedCrop.get_params(
            img, scale=scale, ratio=ratio
        )
        
        # Apply crop with appropriate interpolation
        img_crop = TF.resized_crop(
            img, i, j, h, w, 
            output_size, 
            InterpolationMode.BILINEAR
        )
        depth_crop = TF.resized_crop(
            depth, i, j, h, w, 
            output_size, 
            InterpolationMode.NEAREST  # CRITICAL: nearest for depth to avoid blending
        )
        
        return img_crop, depth_crop
    
    def _to_tensor_pair(self, img, depth, apply_color_aug=True):
        """
        Convert PIL image and depth to tensors.
        
        Args:
            img: PIL Image (RGB)
            depth: PIL Image (depth)
            apply_color_aug: Whether to apply color augmentation
            
        Returns:
            img_tensor: (3, H, W) normalized tensor
            depth_tensor: (1, H, W) tensor in [0, 1]
        """
        # Color augmentation (only RGB, only in training)
        if apply_color_aug and self.split == "train":
            img = self.color_aug(img)
        
        # Convert to tensors
        img_tensor = TF.to_tensor(img)  # (3, H, W), float [0, 1]
        
        # Depth: convert to tensor and normalize
        depth_arr = np.array(depth, dtype=np.float32)
        depth_tensor = torch.from_numpy(depth_arr).unsqueeze(0)  # (1, H, W)
        depth_tensor = depth_tensor / self.depth_max  # Normalize to [0, 1]
        depth_tensor = depth_tensor.clamp(0, 1)
        
        # Normalize RGB with ImageNet stats
        img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        return img_tensor, depth_tensor
    
    def _get_single_view(self, img, depth):
        """Get a single view for simple training or validation."""
        if self.split == "train":
            # Synchronized random crop
            img, depth = self._synchronized_crop(
                img, depth, 
                output_size=self.global_img_size,
                scale=(0.5, 1.0)
            )
            
            # Random horizontal flip (synchronized)
            if random.random() > 0.5:
                img = TF.hflip(img)
                depth = TF.hflip(depth)
        else:
            # Deterministic center crop for validation
            img = TF.resize(img, 256, InterpolationMode.BILINEAR)
            depth = TF.resize(depth, 256, InterpolationMode.NEAREST)
            img = TF.center_crop(img, self.global_img_size)
            depth = TF.center_crop(depth, self.global_img_size)
        
        return self._to_tensor_pair(img, depth)
    
    def _get_multi_view(self, img, depth):
        """
        Get multiple views for JEPA training.
        
        Returns:
            img_views: List of (3, H, W) tensors (global views first, then local)
            depth_views: List of (1, H, W) tensors (matching crops)
        """
        img_views = []
        depth_views = []
        
        # Global views (larger crops, 224x224)
        for _ in range(self.V_global):
            img_crop, depth_crop = self._synchronized_crop(
                img, depth,
                output_size=self.global_img_size,
                scale=(0.4, 1.0),  # Larger scale for global views
                ratio=(0.75, 1.33)
            )
            
            # Random horizontal flip
            if random.random() > 0.5:
                img_crop = TF.hflip(img_crop)
                depth_crop = TF.hflip(depth_crop)
            
            img_t, depth_t = self._to_tensor_pair(img_crop, depth_crop)
            img_views.append(img_t)
            depth_views.append(depth_t)
        
        # Local views (smaller crops, 96x96)
        for _ in range(self.V_local):
            img_crop, depth_crop = self._synchronized_crop(
                img, depth,
                output_size=self.local_img_size,
                scale=(0.05, 0.4),  # Smaller scale for local views
                ratio=(0.75, 1.33)
            )
            
            # Random horizontal flip
            if random.random() > 0.5:
                img_crop = TF.hflip(img_crop)
                depth_crop = TF.hflip(depth_crop)
            
            img_t, depth_t = self._to_tensor_pair(img_crop, depth_crop)
            img_views.append(img_t)
            depth_views.append(depth_t)
        
        return img_views, depth_views
    
    def __getitem__(self, idx):
        img_path, depth_path = self.samples[idx]
        
        # Load images
        img = Image.open(img_path).convert("RGB")
        depth = Image.open(depth_path)  # Keep as-is (mode 'I' = 32-bit int)
        
        if self.multi_view and self.split == "train":
            # Multi-view mode for JEPA training
            return self._get_multi_view(img, depth)
        else:
            # Single view mode for simple training or validation
            return self._get_single_view(img, depth)
    
    def __len__(self):
        return len(self.samples)


def collate_depth(batch):
    """
    Collate function for single-view depth dataset.
    
    Returns:
        images: (B, 3, H, W)
        depths: (B, 1, H, W)
    """
    images = torch.stack([b[0] for b in batch])
    depths = torch.stack([b[1] for b in batch])
    return images, depths


def collate_depth_multiview(batch):
    """
    Collate function for multi-view depth dataset (JEPA training).
    
    Groups views by resolution for efficient batched processing.
    
    Args:
        batch: List of (img_views, depth_views) where each is a list of tensors
        
    Returns:
        img_views_stacked: List of (B, 3, H, W) tensors, grouped by resolution
        depth_views_stacked: List of (B, 1, H, W) tensors, matching img_views
    """
    # Collect views grouped by size
    img_by_size = {}
    depth_by_size = {}
    
    for img_views, depth_views in batch:
        for img_v, depth_v in zip(img_views, depth_views):
            size = img_v.shape[-1]  # Use width as key
            if size not in img_by_size:
                img_by_size[size] = []
                depth_by_size[size] = []
            img_by_size[size].append(img_v)
            depth_by_size[size].append(depth_v)
    
    # Stack each size group and organize by size (descending = global first)
    batch_size = len(batch)
    img_views_stacked = []
    depth_views_stacked = []
    
    for size in sorted(img_by_size.keys(), reverse=True):
        # Stack all views of this size: (num_views_total, C, H, W)
        imgs_stacked = torch.stack(img_by_size[size])
        depths_stacked = torch.stack(depth_by_size[size])
        
        # Reshape to (B, num_views_per_sample, C, H, W)
        num_views = len(img_by_size[size]) // batch_size
        imgs_stacked = imgs_stacked.reshape(batch_size, num_views, *imgs_stacked.shape[1:])
        depths_stacked = depths_stacked.reshape(batch_size, num_views, *depths_stacked.shape[1:])
        
        # Add each view position as separate tensor
        for v_idx in range(num_views):
            img_views_stacked.append(imgs_stacked[:, v_idx])  # (B, 3, H, W)
            depth_views_stacked.append(depths_stacked[:, v_idx])  # (B, 1, H, W)
    
    return img_views_stacked, depth_views_stacked


# Quick test
if __name__ == "__main__":
    print("=== Single View Mode ===")
    ds_single = DepthDataset(
        split="train", 
        global_img_size=224,
        neighborhoods=[0, 1, 2],
        multi_view=False
    )
    print(f"Dataset size: {len(ds_single)}")
    
    img, depth = ds_single[0]
    print(f"Image shape: {img.shape}, dtype: {img.dtype}")
    print(f"Depth shape: {depth.shape}, dtype: {depth.dtype}")
    print(f"Depth range: [{depth.min():.4f}, {depth.max():.4f}]")
    
    print("\n=== Multi View Mode (JEPA) ===")
    ds_multi = DepthDataset(
        split="train",
        global_img_size=224,
        local_img_size=96,
        neighborhoods=[0, 1, 2],
        V_global=2,
        V_local=4,
        multi_view=True
    )
    print(f"Dataset size: {len(ds_multi)}")
    
    img_views, depth_views = ds_multi[0]
    print(f"Number of views: {len(img_views)} (2 global + 4 local)")
    for i, (img_v, depth_v) in enumerate(zip(img_views, depth_views)):
        print(f"  View {i}: img {img_v.shape}, depth {depth_v.shape}")
    
    print("\n=== Test Collate ===")
    from torch.utils.data import DataLoader
    loader = DataLoader(ds_multi, batch_size=4, collate_fn=collate_depth_multiview)
    img_batch, depth_batch = next(iter(loader))
    print(f"Batch: {len(img_batch)} view groups")
    for i, (img_v, depth_v) in enumerate(zip(img_batch, depth_batch)):
        print(f"  View {i}: img {img_v.shape}, depth {depth_v.shape}")