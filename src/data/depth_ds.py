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
import torch.nn.functional as F
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
        data_root=f"{os.getcwd()}/datalink/cache/datasets--benediktkol--DDOS/snapshots/1ed1314d32ef3a5a7e1434000783a8433517bd0e/data/",
        depth_max=65535.0,   # Max depth value for normalization
        V_global=2,          # Number of global views (for JEPA)
        V_local=4,           # Number of local views (for JEPA)
        multi_view=True,     # Enable multi-view for JEPA training
    ):
        self.split = split
        self.global_img_size = global_img_size
        self.local_img_size = local_img_size
        self.depth_max = depth_max
        self.data_root = os.path.join(
            os.getcwd(),
            "datalink",
            "cache/datasets--benediktkol--DDOS/snapshots/1ed1314d32ef3a5a7e1434000783a8433517bd0e/data/"
        )
        self.V_global = V_global
        self.V_local = V_local
        self.multi_view = multi_view
        
        # Collect all image/depth pairs
        # Split train/val (95/5)
        self.samples = self._collect_samples()
        rng = random.Random(42)
        indices = list(range(len(self.samples)))
        rng.shuffle(indices)
        split_idx = int(0.95 * len(indices))
        if split == "train":
            self.samples = [self.samples[i] for i in indices[:split_idx]]
        elif split == "val":
            self.samples = [self.samples[i] for i in indices[split_idx:]]
        
        
        
        
        print(f"DepthDataset [{split}]: {len(self.samples)} samples, "
              f"multi_view={multi_view}, V_global={V_global}, V_local={V_local}")
        
        # Color augmentation (applied only to RGB, NOT depth)
        self.color_aug = v2.Compose([
            v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            v2.RandomGrayscale(p=0.1),
        ])
    
    def _collect_samples(self):
        """Gather all (image_path, depth_path) pairs."""
        samples = []
        
        # Get all neighborhood folders
        dir_split = "train" if self.split == "val" else self.split
        data_dir = f"{self.data_root}{dir_split}/neighbourhood"
        
        folders = [d for d in os.listdir(data_dir) 
                    if os.path.isdir(os.path.join(data_dir, d)) and d.isdigit()]
        # print(folders)
        
        for folder in folders:
            img_dir = os.path.join(data_dir, folder, "image")
            depth_dir = os.path.join(data_dir, folder, "depth")
            # print(img_dir)
            if not os.path.isdir(img_dir) or not os.path.isdir(depth_dir):
                continue
            
            # Get all image files (symlinks)
            for img_name in os.listdir(img_dir):
                if not img_name.endswith('.png'):
                    continue
                img_file = os.path.join(img_dir, img_name)
                depth_file = os.path.join(depth_dir, img_name)

                if img_file and depth_file:
                    # print (img_file)
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
        """
        Get a single view.
        
        Train: Random crop + flip
        Val/Test: Grid of non-overlapping patches covering the full image
        """
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
                
            return self._to_tensor_pair(img, depth)
            
        else:
            # Deterministic patching for validation / test
            patch_size = self.global_img_size
            
            # Convert to tensors first 
            # Note: _to_tensor_pair does normalization, so we do it manually here 
            # to handle the full image before patching
            
            # RGB normalization constants
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            
            # Prepare Image
            img_t = TF.to_tensor(img) # (3, H, W)
            img_t = TF.normalize(img_t, mean=mean, std=std)
            
            # Prepare Depth
            depth_arr = np.array(depth, dtype=np.float32)
            depth_t = torch.from_numpy(depth_arr).unsqueeze(0) # (1, H, W)
            depth_t = depth_t / self.depth_max
            depth_t = depth_t.clamp(0, 1)
            
            # Pad to multiple of patch_size
            _, H, W = img_t.shape
            pad_h = (patch_size - H % patch_size) % patch_size
            pad_w = (patch_size - W % patch_size) % patch_size
            
            if pad_h > 0 or pad_w > 0:
                img_t = F.pad(img_t, (0, pad_w, 0, pad_h), mode='reflect')
                depth_t = F.pad(depth_t, (0, pad_w, 0, pad_h), mode='replicate')
                
            # Unfold into patches
            # Input to unfold must be (N, C, H, W)
            img_t = img_t.unsqueeze(0)
            depth_t = depth_t.unsqueeze(0)
            
            # (1, C*P*P, L) where L is number of patches
            img_patches = F.unfold(img_t, kernel_size=patch_size, stride=patch_size)
            depth_patches = F.unfold(depth_t, kernel_size=patch_size, stride=patch_size)
            
            # Reshape to (L, C, P, P)
            # 1. Transpose to (1, L, C*P*P) -> squeeze to (L, C*P*P)
            # 2. View as (L, C, P, P)
            L = img_patches.size(2)
            
            img_patches = img_patches.transpose(1, 2).squeeze(0)
            img_patches = img_patches.view(L, 3, patch_size, patch_size)
            
            depth_patches = depth_patches.transpose(1, 2).squeeze(0)
            depth_patches = depth_patches.view(L, 1, patch_size, patch_size)
            
            return img_patches, depth_patches, (H, W)
    
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
    Handles both single tensors (Training) and stacks of patches (Validation).
    
    Args:
        batch: List of (img, depth)
        
    Returns:
        images: (B_total, 3, H, W)
        depths: (B_total, 1, H, W)
        patch_counts: List[int] or None. Number of patches per original image. 
                      None for single-view training.
    """
    # Check if elements are 3D tensors (single view) or 4D tensors (patches)
    elem_img = batch[0][0]
    
    if elem_img.ndim == 3: # (C, H, W) - Training / Single View
        images = torch.stack([b[0] for b in batch])
        depths = torch.stack([b[1] for b in batch])
        patch_counts = None
        shapes = None
    else: # (N_patches, C, H, W) - Validation / Patches
        # Concatenate all patches from all images in the batch
        images = torch.cat([b[0] for b in batch], dim=0)
        depths = torch.cat([b[1] for b in batch], dim=0)
        patch_counts = [b[0].shape[0] for b in batch]
        shapes = [b[2] for b in batch]
        
    return images, depths, patch_counts, shapes


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


class DepthDatasetFull(DepthDataset):
    """
    Dataset for full-image training (adaptive sizes) with ResNet.
    
    Resizes images to be divisible by 32 (stride of ResNet50)
    while maintaining aspect ratio, up to max_img_size.
    """
    
    def __init__(
        self, 
        split="train",
        max_img_size=1536, # Multiple of 32 close to 1500
        data_root=None, # Use default from parent if None
        depth_max=65535.0,
        multi_view=True, # If True, returns [img], [depth] lists for compatibility
    ):
        # Initialize parent to get file lists
        super().__init__(
            split=split,
            data_root=data_root if data_root else f"{os.getcwd()}/datalink/cache/datasets--benediktkol--DDOS/snapshots/1ed1314d32ef3a5a7e1434000783a8433517bd0e/data/",
            depth_max=depth_max,
            multi_view=multi_view
        )
        self.max_img_size = max_img_size
        
        print(f"DepthDatasetFull [{split}]: {len(self.samples)} samples, max_size={max_img_size}")
        
    def _resize_adaptive(self, img, depth):
        """
        Resize to max_img_size while keeping aspect ratio and ensuring % 32 == 0.
        """
        w, h = img.size
        
        # Scale to max dimension
        scale = min(self.max_img_size / w, self.max_img_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Round to nearest multiple of 32
        new_w = round(new_w / 32) * 32
        new_h = round(new_h / 32) * 32
        
        # Resize
        img = img.resize((new_w, new_h), Image.BILINEAR)
        depth = depth.resize((new_w, new_h), Image.NEAREST)
        
        return img, depth

    def __getitem__(self, idx):
        img_path, depth_path = self.samples[idx]
        
        img = Image.open(img_path).convert("RGB")
        depth = Image.open(depth_path)
        
        # Resize
        img, depth = self._resize_adaptive(img, depth)
        
        # To Tensor
        img_t, depth_t = self._to_tensor_pair(img, depth)
        
        if self.multi_view:
            # Return as list of views (length 1) for compatibility
            return [img_t], [depth_t]
        else:
            return img_t, depth_t



# Quick test
if __name__ == "__main__":
    print("=== Single View Mode ===")
    ds_single = DepthDataset(
        split="test", 
        global_img_size=224,
        # neighborhoods=[0, 1, 2],
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
        # neighborhoods=[0, 1, 2],
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