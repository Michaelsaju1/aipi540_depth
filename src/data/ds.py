import torch
# import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import v2
from datasets import load_dataset
import random
import os
import glob
from PIL import Image

class HFDataset(Dataset):
    """
    Dataset for JEPA training with multi-view augmentation.
    Supports multiple dataset types: inet100, cifar10, neighbourhood, ddos.
    """
    def __init__(self, split, V_global=2, V_local=4, device="cuda", global_img_size=224, local_img_size=96, dataset="inet100"):
        self.V_global = V_global
        self.V_local = V_local
        self.split = split
        self.global_img_size = global_img_size
        self.local_img_size = local_img_size
        self.dataset_name = dataset
        
        self._get_ds(dataset)
        
        # 2. Define Transforms
        # Global Views: 224x224
        self.global_transform = v2.Compose([
            v2.RandomResizedCrop(self.global_img_size, scale=(0.08, 1.0)),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
        ])
        
        # Local Views: 96x96
        self.local_transform = v2.Compose([
            v2.RandomResizedCrop(self.local_img_size, scale=(0.05, 0.4)),
            v2.ToImage(),
        ])

        # Test transform
        self.test_transform = v2.Compose([
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.bfloat16, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _get_ds(self, dataset):
        """
        Load dataset based on type.
        Supports: inet100, cifar10, neighbourhood (local), ddos
        """
        # DDOS dataset from HuggingFace cache
        # Path: datalink/cache/datasets--benediktkol--DDOS/.../data/{train,test}/neighbourhood/{0-N}
        self.is_local = True
        
        base_dir = os.path.join(
            os.getcwd(), 
            "datalink", 
            "cache/datasets--benediktkol--DDOS/snapshots/1ed1314d32ef3a5a7e1434000783a8433517bd0e/data",
            self.split,
            "neighbourhood"
        )
        
        if not os.path.exists(base_dir):
            raise ValueError(f"DDOS dataset directory not found: {base_dir}")
        
        # Get all scene directories
        scene_dirs = sorted([d for d in os.listdir(base_dir) 
                            if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()])
        
        # Collect all image files from all scenes
        self.image_files = []
        for scene_dir in scene_dirs:
            scene_path = os.path.join(base_dir, scene_dir, "image")
            if os.path.exists(scene_path):
                images = sorted(glob.glob(os.path.join(scene_path, "*.png")))
                self.image_files.extend(images)
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in DDOS dataset: {base_dir}")
        
        print(f"Loaded {len(self.image_files)} images from DDOS {self.split} split across {len(scene_dirs)} scenes")
        


    def _load_image(self, entry):
        """Helper to handle safe image extraction from row entry."""
        if isinstance(entry, str):
            # Local file path
            return Image.open(entry).convert("RGB")
        elif "image" in entry:
            return entry["image"]  # PIL Image already returned by HF dataset
        elif "img" in entry:
            return entry["img"].convert("RGB")
        else:
            raise ValueError("Image not found in entry")

    def __getitem__(self, i):
        # Load from local file
        img_path = self.image_files[i]
        img = Image.open(img_path).convert("RGB")
        label = 0  # Dummy label for self-supervised learning

        if self.split == 'train':
            views = []
            
            # Global Views
            if self.V_global > 0:
                views += [self.global_transform(img) for _ in range(self.V_global)]
            
            # Local Views
            if self.V_local > 0:
                views += [self.local_transform(img) for _ in range(self.V_local)]
            
            return views, label
        else:
            # Validation/Test
            return [self.test_transform(img)], label

    def __len__(self):
        if self.is_local:
            return len(self.image_files)
        else:
            return len(self.ds)


class CrossInstanceDataset(Dataset):
    """
    Wrapper dataset that creates cross-instance pairs for contrastive learning.
    Each sample returns views from two different random images.
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.length = len(base_dataset)
    
    def __getitem__(self, idx):
        # Get first sample
        views1, label1 = self.base_dataset[idx]
        
        # Get random second sample (different from first)
        idx2 = random.randint(0, self.length - 1)
        while idx2 == idx:
            idx2 = random.randint(0, self.length - 1)
        views2, label2 = self.base_dataset[idx2]
        
        # Combine views from both samples
        all_views = views1 + views2
        return all_views, (label1, label2)
    
    def __len__(self):
        return self.length


def collate_views(batch):
    """
    Custom collate function for multi-view batches.
    
    Input: List of (views, label) tuples where views is a list of tensors
    Output: Stacked views tensor and labels tensor
    
    Efficiently stacks views by grouping same-resolution views together.
    """
    if len(batch) == 0:
        return None, None
    
    # Separate views and labels
    all_views = [item[0] for item in batch]  # List of lists
    all_labels = [item[1] for item in batch]
    
    # Number of views per sample
    n_views = len(all_views[0])
    batch_size = len(batch)
    
    # Stack views: (batch_size, n_views, C, H, W) -> (batch_size * n_views, C, H, W)
    # Group by view index for efficient processing
    stacked_views = []
    for view_idx in range(n_views):
        view_batch = torch.stack([all_views[i][view_idx] for i in range(batch_size)])
        stacked_views.append(view_batch)
    
    # Concatenate all views: [(B, C, H1, W1), (B, C, H1, W1), (B, C, H2, W2), ...]
    # Result: List of tensors, grouped by resolution
    
    # Convert labels to tensor
    if isinstance(all_labels[0], tuple):
        # Cross-instance labels
        labels = torch.tensor([[l[0], l[1]] for l in all_labels])
    else:
        labels = torch.tensor(all_labels)
    
    return stacked_views, labels


class MultiTaskDataset(Dataset):
    """
    Multi-task dataset for joint depth estimation and segmentation.
    
    Loads triplets of (image, depth, segmentation) from the neighbourhood/DDOS dataset structure.
    Each scene folder contains: image/, depth/, segmentation/ subdirectories with aligned frames.
    
    Args:
        split: 'train' or 'test'
        dataset: 'neighbourhood' (local scenes) or 'ddos' (HF cached dataset)
        scene_id: For neighbourhood, specific scene number (0-14), or None for all scenes
        img_size: Target image size for resizing (default 224)
        augment: Whether to apply data augmentation (only for train split)
    """
    def __init__(self, split="train", dataset="neighbourhood", scene_id=None, img_size=224, augment=True):
        self.split = split
        self.dataset = dataset
        self.img_size = img_size
        self.augment = augment and (split == "train")
        
        # Collect all triplets (image, depth, segmentation paths)
        self.data_triplets = self._collect_data(dataset, scene_id)
        
        if len(self.data_triplets) == 0:
            raise ValueError(f"No data found for {dataset} {split} split")
        
        print(f"Loaded {len(self.data_triplets)} samples for {dataset} {split} split")
        
        # Define transforms
        self._setup_transforms()
    
    def _collect_data(self, dataset, scene_id):
        """Collect all (image, depth, segmentation) triplets."""
        triplets = []
        
        
        # DDOS dataset from HuggingFace cache
        base_dir = os.path.join(
            os.getcwd(),
            "datalink",
            "cache/datasets--benediktkol--DDOS/snapshots/1ed1314d32ef3a5a7e1434000783a8433517bd0e/data",
            self.split,
            "neighbourhood"
        )
        
        if not os.path.exists(base_dir):
            raise ValueError(f"DDOS dataset directory not found: {base_dir}")
        
        scene_dirs = sorted([d for d in os.listdir(base_dir)
                            if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()])
        
        for scene_dir in scene_dirs:
            scene_path = os.path.join(base_dir, scene_dir)
            triplets.extend(self._collect_scene_triplets(scene_path))
    
        return triplets
    
    def _collect_scene_triplets(self, scene_path):
        """Collect triplets from a single scene directory."""
        triplets = []
        
        image_dir = os.path.join(scene_path, "image")
        depth_dir = os.path.join(scene_path, "depth")
        seg_dir = os.path.join(scene_path, "segmentation")
        
        # Check all required directories exist
        if not all(os.path.exists(d) for d in [image_dir, depth_dir, seg_dir]):
            print(f"Warning: Skipping {scene_path} - missing required subdirectories")
            return triplets
        
        # Get all image files
        image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        
        for img_path in image_files:
            # Get corresponding depth and segmentation paths
            filename = os.path.basename(img_path)
            depth_path = os.path.join(depth_dir, filename)
            seg_path = os.path.join(seg_dir, filename)
            
            # Only include if all three files exist
            if os.path.exists(depth_path) and os.path.exists(seg_path):
                triplets.append({
                    'image': img_path,
                    'depth': depth_path,
                    'segmentation': seg_path
                })
        
        return triplets
    
    def _setup_transforms(self):
        """Setup synchronized transforms for training and testing."""
        if self.augment:
            # Training: Synchronized geometric transforms
            self.geometric_transform = v2.Compose([
                v2.Resize((self.img_size, self.img_size)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomResizedCrop(self.img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            ])
            
            # Image-only transforms (color jitter, normalization)
            self.img_only_transform = v2.Compose([
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            # Depth-only transforms (no color jitter, just conversion)
            self.depth_only_transform = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ])
            
            # Segmentation-only transforms (nearest neighbor for resize)
            self.seg_only_transform = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.long, scale=False),
            ])
            
        else:
            # Test: Simple resize without augmentation
            self.geometric_transform = v2.Compose([
                v2.Resize((self.img_size, self.img_size)),
            ])
            
            self.img_only_transform = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            self.depth_only_transform = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ])
            
            self.seg_only_transform = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.long, scale=False),
            ])
    
    def __getitem__(self, idx):
        """
        Returns:
            image: (3, H, W) normalized RGB image
            depth: (1, H, W) depth map
            segmentation: (1, H, W) segmentation mask
        """
        triplet = self.data_triplets[idx]
        
        # Load images as PIL
        image = Image.open(triplet['image']).convert('RGB')
        depth = Image.open(triplet['depth']).convert('L')  # Grayscale
        segmentation = Image.open(triplet['segmentation']).convert('L')  # Grayscale
        
        # Apply SYNCHRONIZED geometric transforms to all three at once
        # torchvision v2 supports applying the same transform to multiple images
        if self.augment:
            # Create a combined batch for synchronized transforms
            # Stack them together so they get the same random parameters
            stacked = torch.stack([
                v2.ToImage()(image),
                v2.ToImage()(depth),
                v2.ToImage()(segmentation)
            ])
            
            # Apply geometric transforms to all at once (same crop/flip)
            # Need to handle segmentation with nearest neighbor interpolation
            image_pil, depth_pil, seg_pil = image, depth, segmentation
            
            # Get random parameters for transforms
            # Apply RandomResizedCrop with same parameters
            i, j, h, w = v2.RandomResizedCrop.get_params(
                image_pil, 
                scale=(0.8, 1.0), 
                ratio=(0.9, 1.1)
            )
            
            # Apply same crop to all three
            image_pil = v2.functional.crop(image_pil, i, j, h, w)
            depth_pil = v2.functional.crop(depth_pil, i, j, h, w)
            seg_pil = v2.functional.crop(seg_pil, i, j, h, w)
            
            # Resize to target size (use bilinear for image/depth, nearest for seg)
            image_pil = v2.functional.resize(image_pil, (self.img_size, self.img_size), 
                                            interpolation=v2.InterpolationMode.BILINEAR)
            depth_pil = v2.functional.resize(depth_pil, (self.img_size, self.img_size), 
                                            interpolation=v2.InterpolationMode.BILINEAR)
            seg_pil = v2.functional.resize(seg_pil, (self.img_size, self.img_size), 
                                          interpolation=v2.InterpolationMode.NEAREST)
            
            # Apply horizontal flip with same probability
            if torch.rand(1) < 0.5:
                image_pil = v2.functional.hflip(image_pil)
                depth_pil = v2.functional.hflip(depth_pil)
                seg_pil = v2.functional.hflip(seg_pil)
            
            # Now apply modality-specific transforms
            image = self.img_only_transform(image_pil)
            depth = self.depth_only_transform(depth_pil)
            segmentation = self.seg_only_transform(seg_pil)
            
        else:
            # Test mode: simple resize
            image_pil = v2.functional.resize(image, (self.img_size, self.img_size), 
                                            interpolation=v2.InterpolationMode.BILINEAR)
            depth_pil = v2.functional.resize(depth, (self.img_size, self.img_size), 
                                            interpolation=v2.InterpolationMode.BILINEAR)
            seg_pil = v2.functional.resize(segmentation, (self.img_size, self.img_size), 
                                          interpolation=v2.InterpolationMode.NEAREST)
            
            image = self.img_only_transform(image_pil)
            depth = self.depth_only_transform(depth_pil)
            segmentation = self.seg_only_transform(seg_pil)
        
        # Ensure depth and segmentation have channel dimension
        if depth.dim() == 2:
            depth = depth.unsqueeze(0)
        if segmentation.dim() == 2:
            segmentation = segmentation.unsqueeze(0)
        
        return {
            'image': image,
            'depth': depth,
            'segmentation': segmentation,
            'image_path': triplet['image']
        }
    
    def __len__(self):
        return len(self.data_triplets)


def collate_multitask(batch):
    """
    Custom collate function for multi-task batch.
    
    Input: List of dicts with keys: image, depth, segmentation, image_path
    Output: Batched tensors
    """
    images = torch.stack([item['image'] for item in batch])
    depths = torch.stack([item['depth'] for item in batch])
    segmentations = torch.stack([item['segmentation'] for item in batch])
    paths = [item['image_path'] for item in batch]
    
    return {
        'image': images,
        'depth': depths,
        'segmentation': segmentations,
        'image_path': paths
    }


# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    print(f"Current Working Directory: {os.getcwd()}")
    print("=" * 60)
    
    # Test 1: Local neighbourhood dataset (single scene)
    print("\n--- Test 1: Local Neighbourhood Dataset (scene 13) ---")
    try:
        train_ds = HFDataset(split="train", dataset="local_13", V_global=2, V_local=4)
        print(f"✓ Train Dataset Size: {len(train_ds)}")
        
        test_ds = HFDataset(split="test", dataset="local_13", V_global=2, V_local=4)
        print(f"✓ Test Dataset Size: {len(test_ds)}")
        
        if len(train_ds) > 0:
            views, label = train_ds[0]
            print(f"✓ Successfully loaded training sample")
            print(f"  - Number of views: {len(views)} (2 global + 4 local)")
            print(f"  - Global view shape: {views[0].shape}")
            print(f"  - Local view shape: {views[2].shape}")
            print(f"  - Label: {label}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 2: DDOS dataset (multiple scenes)
    print("\n--- Test 2: DDOS Dataset (all scenes) ---")
    try:
        train_ds = HFDataset(split="train", dataset="ddos", V_global=2, V_local=4)
        print(f"✓ Train Dataset Size: {len(train_ds)}")
        
        test_ds = HFDataset(split="test", dataset="ddos", V_global=2, V_local=4)
        print(f"✓ Test Dataset Size: {len(test_ds)}")
        
        if len(train_ds) > 0:
            views, label = train_ds[0]
            print(f"✓ Successfully loaded training sample")
            print(f"  - Number of views: {len(views)}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 3: CrossInstanceDataset wrapper
    print("\n--- Test 3: CrossInstanceDataset Wrapper ---")
    try:
        base_ds = HFDataset(split="train", dataset="local_0", V_global=2, V_local=4)
        cross_ds = CrossInstanceDataset(base_ds)
        print(f"✓ CrossInstance Dataset Size: {len(cross_ds)}")
        
        if len(cross_ds) > 0:
            views, labels = cross_ds[0]
            print(f"✓ Successfully loaded cross-instance sample")
            print(f"  - Number of views: {len(views)} (should be 12: 6 from each image)")
            print(f"  - Labels: {labels} (tuple of two labels)")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 4: Collate function
    print("\n--- Test 4: Collate Function ---")
    try:
        from torch.utils.data import DataLoader
        ds = HFDataset(split="train", dataset="local_0", V_global=2, V_local=4)
        loader = DataLoader(ds, batch_size=4, collate_fn=collate_views)
        
        batch = next(iter(loader))
        views, labels = batch
        print(f"✓ Successfully created batch with collate_views")
        print(f"  - Number of view groups: {len(views)}")
        print(f"  - First view group shape: {views[0].shape}")
        print(f"  - Labels shape: {labels.shape}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print("Testing complete!")
    
    # Test 5: MultiTaskDataset (Depth + Segmentation)
    print("\n--- Test 5: MultiTaskDataset (Depth + Segmentation) ---")
    try:
        # Test with neighbourhood dataset
        train_ds = MultiTaskDataset(split="train", dataset="neighbourhood", scene_id=0, img_size=224)
        print(f"✓ Train Dataset Size: {len(train_ds)}")
        
        if len(train_ds) > 0:
            sample = train_ds[0]
            print(f"✓ Successfully loaded multi-task sample")
            print(f"  - Image shape: {sample['image'].shape}")
            print(f"  - Depth shape: {sample['depth'].shape}")
            print(f"  - Segmentation shape: {sample['segmentation'].shape}")
            print(f"  - Image range: [{sample['image'].min():.2f}, {sample['image'].max():.2f}]")
            print(f"  - Depth range: [{sample['depth'].min():.2f}, {sample['depth'].max():.2f}]")
            print(f"  - Segmentation unique values: {sample['segmentation'].unique().tolist()[:10]}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 6: MultiTaskDataset with DataLoader
    print("\n--- Test 6: MultiTaskDataset with DataLoader ---")
    try:
        from torch.utils.data import DataLoader
        ds = MultiTaskDataset(split="train", dataset="neighbourhood", scene_id=0, img_size=224)
        loader = DataLoader(ds, batch_size=4, collate_fn=collate_multitask, num_workers=0)
        
        batch = next(iter(loader))
        print(f"✓ Successfully created multi-task batch")
        print(f"  - Batch images shape: {batch['image'].shape}")
        print(f"  - Batch depths shape: {batch['depth'].shape}")
        print(f"  - Batch segmentations shape: {batch['segmentation'].shape}")
        print(f"  - Number of paths: {len(batch['image_path'])}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 7: DDOS MultiTaskDataset
    print("\n--- Test 7: DDOS MultiTaskDataset ---")
    try:
        train_ds = MultiTaskDataset(split="train", dataset="ddos", img_size=224)
        print(f"✓ Train Dataset Size: {len(train_ds)}")
        
        test_ds = MultiTaskDataset(split="test", dataset="ddos", img_size=224)
        print(f"✓ Test Dataset Size: {len(test_ds)}")
        
        if len(train_ds) > 0:
            sample = train_ds[0]
            print(f"✓ Successfully loaded DDOS multi-task sample")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print("All tests complete!")