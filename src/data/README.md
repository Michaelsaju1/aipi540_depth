# Dataset Documentation

This directory contains dataset implementations for the AIPI-540 CV Hackathon project.

## Available Datasets

### 1. HFDataset - Self-Supervised Learning
Multi-view dataset for JEPA (Joint Embedding Predictive Architecture) training.

**Supported Sources:**
- `inet100`: ImageNet-100 subset
- `cifar10`: CIFAR-10 dataset
- `neighbourhood` or `local_X`: Local neighbourhood scenes (where X is scene number 0-14)
- `ddos`: DDOS dataset from HuggingFace

**Features:**
- Generates multiple augmented views per image
- Configurable global views (224x224) and local views (96x96)
- Supports cross-instance pairs for contrastive learning

**Usage:**
```python
from src.data.ds import HFDataset, collate_views
from torch.utils.data import DataLoader

# Single scene
train_ds = HFDataset(
    split="train",
    dataset="local_0",
    V_global=2,      # Number of global views
    V_local=4,       # Number of local views
    global_img_size=224,
    local_img_size=96
)

# All DDOS scenes
train_ds = HFDataset(split="train", dataset="ddos", V_global=2, V_local=4)

# Create DataLoader
loader = DataLoader(train_ds, batch_size=32, collate_fn=collate_views)

for views, labels in loader:
    # views is a list of tensors grouped by resolution
    # views[0]: global view 1 (B, 3, 224, 224)
    # views[1]: global view 2 (B, 3, 224, 224)
    # views[2:6]: local views (B, 3, 96, 96)
    pass
```

### 2. MultiTaskDataset - Supervised Multi-Task Learning
Dataset for joint depth estimation and semantic segmentation.

**Supported Sources:**
- `neighbourhood`: Local neighbourhood scenes at `datalink/neighbourhood/{0-14}/`
- `ddos`: DDOS dataset from HuggingFace cache

**Structure:**
Each scene contains aligned triplets:
```
scene_X/
├── image/         # RGB images
├── depth/         # Depth maps
└── segmentation/  # Segmentation masks
```

**Features:**
- Loads aligned image, depth, and segmentation triplets
- Synchronized augmentation for train split
- Configurable image size
- Automatic scene discovery

**Usage:**
```python
from src.data.ds import MultiTaskDataset, collate_multitask
from torch.utils.data import DataLoader

# Single scene
train_ds = MultiTaskDataset(
    split="train",
    dataset="neighbourhood",
    scene_id=0,      # Specific scene, or None for all
    img_size=224,
    augment=True     # Apply data augmentation
)

# All DDOS scenes
train_ds = MultiTaskDataset(split="train", dataset="ddos", img_size=224)

# Create DataLoader
loader = DataLoader(train_ds, batch_size=16, collate_fn=collate_multitask)

for batch in loader:
    images = batch['image']          # (B, 3, H, W) - normalized RGB
    depths = batch['depth']          # (B, 1, H, W) - depth maps
    segmentations = batch['segmentation']  # (B, 1, H, W) - segmentation masks
    paths = batch['image_path']      # List of image paths
```

### 3. CrossInstanceDataset - Contrastive Learning
Wrapper that creates cross-instance pairs from any base dataset.

**Usage:**
```python
from src.data.ds import HFDataset, CrossInstanceDataset

base_ds = HFDataset(split="train", dataset="local_0")
cross_ds = CrossInstanceDataset(base_ds)

# Each sample returns views from two different images
views, (label1, label2) = cross_ds[0]
```

## Dataset Directory Structure

```
datalink/
├── neighbourhood/           # Local scenes (0-14)
│   ├── 0/
│   │   ├── image/          # RGB images (.png)
│   │   ├── depth/          # Depth maps (.png)
│   │   ├── segmentation/   # Segmentation masks (.png)
│   │   ├── flow/           # Optical flow
│   │   └── metadata.csv
│   ├── 1/
│   └── ...
│
└── cache/
    └── datasets--benediktkol--DDOS/
        └── snapshots/.../data/
            ├── train/
            │   └── neighbourhood/
            │       ├── 0/
            │       ├── 1/
            │       └── ...
            └── test/
                └── neighbourhood/
                    └── ...
```

## Collate Functions

### collate_views
For multi-view batches from HFDataset.
- Groups views by resolution
- Returns list of tensors and labels

### collate_multitask
For multi-task batches from MultiTaskDataset.
- Stacks images, depths, and segmentations
- Returns dictionary with all modalities

## Testing

Run the module directly to test all dataset implementations:

```bash
cd /home/users/aho13/aipi-540-cv-hackathon
python -m src.data.ds
```

This will run comprehensive tests for:
1. HFDataset with local neighbourhood
2. HFDataset with DDOS
3. CrossInstanceDataset wrapper
4. Collate function for multi-view
5. MultiTaskDataset with neighbourhood
6. MultiTaskDataset with DataLoader
7. MultiTaskDataset with DDOS

## Data Preprocessing

### Image Normalization
RGB images are normalized with ImageNet statistics:
- Mean: [0.485, 0.456, 0.406]
- Std: [0.229, 0.224, 0.225]

### Depth Maps
- Loaded as grayscale (single channel)
- Scaled to [0, 1] range
- Shape: (1, H, W)

### Segmentation Masks
- Loaded as grayscale (single channel)
- Values represent class IDs (long tensor)
- Uses nearest neighbor interpolation for resizing
- Shape: (1, H, W)

## Augmentation

### Training Augmentation (MultiTaskDataset)
- Random horizontal flip (p=0.5)
- Color jitter (brightness, contrast, saturation, hue)
- Synchronized across all modalities

### Test Augmentation
- Simple resize and normalization
- No random augmentation

## Tips

1. **Memory Optimization**: Use `num_workers > 0` in DataLoader for parallel data loading
2. **Scene Selection**: Start with a single scene for debugging, then scale to all scenes
3. **Batch Size**: Adjust based on GPU memory (16-32 typical for 224x224 images)
4. **Mixed Precision**: Use `torch.amp.autocast()` for faster training with less memory
