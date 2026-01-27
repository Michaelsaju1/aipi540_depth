# Depth Estimation with JEPA (LeJEPA + SIGReg)

A depth estimation model that combines **Vision Transformer (ViT)** encoding with **JEPA-style multi-view self-supervised learning** and **SIGReg regularization**.

## Overview

This project adapts the LeJEPA (Latent Embedding Joint-Embedding Predictive Architecture) framework for dense depth prediction. Instead of learning representations for image classification, we use JEPA's multi-view consistency objective alongside direct depth supervision.

### Key Components

| File | Description |
|------|-------------|
| `depth_ds.py` | Dataset with synchronized RGB/depth transforms and multi-view generation |
| `depth_model.py` | ViT encoder + convolutional decoder for dense depth prediction |
| `train_depth_jepa.py` | Training loop with LeJEPA loss + depth supervision |
| `loss.py` | SIGReg regularization module (shared with classification JEPA) |

## Architecture

```
RGB Image (B, 3, H, W)
        │
        ▼
┌─────────────────────────────┐
│     ViT Encoder             │  ← Pretrained vit_small_patch16_224
│  (dynamic_img_size=True)    │
└─────────────────────────────┘
        │
        ├──────────────────────────┐
        ▼                          ▼
┌───────────────┐          ┌──────────────┐
│  Patch Tokens │          │  CLS Token   │
│ (B, 196, 384) │          │  (B, 384)    │
└───────────────┘          └──────────────┘
        │                          │
        ▼                          ▼
┌───────────────┐          ┌──────────────┐
│    Decoder    │          │  Projection  │
│  14→28→56→    │          │   Head       │
│  112→224      │          │ 384→2048→512 │
└───────────────┘          └──────────────┘
        │                          │
        ▼                          ▼
   Depth Map                  Embedding
  (B, 1, H, W)               (B, 512)
                                   │
                                   ▼
                            LeJEPA Loss
                           + SIGReg
```

## JEPA Loss Function

The LeJEPA loss encourages **view-invariant representations** while SIGReg ensures embeddings follow a **Gaussian distribution** (preventing collapse).

### Loss Components

```python
def LeJEPA_Depth(global_proj, all_proj, sigreg, lamb):
    """
    global_proj: (N, Vg, D) - Embeddings from global views (224×224)
    all_proj: (N, V, D) - Embeddings from all views (global + local)
    """
    # 1. Center = mean of global view embeddings
    centers = global_proj.mean(dim=1, keepdim=True)  # (N, 1, D)
    
    # 2. Prediction loss: all views should match the center
    sim_loss = (centers - all_proj).square().mean()
    
    # 3. SIGReg: ensure embeddings are Gaussian (prevents collapse)
    sigreg_loss = mean([sigreg(all_proj[:, i, :]) for i in range(V)])
    
    # 4. Combined
    return (1 - lamb) * sim_loss + lamb * sigreg_loss
```

### Total Training Loss

```python
# Depth supervision on all views
depth_loss = mean([ScaleInvariantLoss(pred, target) for each view])

# JEPA embedding consistency
jepa_loss = LeJEPA_Depth(global_emb, all_emb, sigreg, lamb=0.05)

# Combined
total_loss = depth_weight * depth_loss + jepa_weight * jepa_loss
```

## Dataset: DDOS (Depth from Driving Open Scenes)

**Source**: [benediktkol/DDOS](https://huggingface.co/datasets/benediktkol/DDOS)

### Data Structure

```
datalink/neighbourhood/
├── 0/
│   ├── image/     # RGB images (1280×720, PNG)
│   │   ├── 0.png → blob/HASH
│   │   └── ...
│   └── depth/     # Depth maps (1280×720, 32-bit int)
│       ├── 0.png → blob/HASH
│       └── ...
├── 1/
└── ...
```

### Depth Format

- **Resolution**: 1280 × 720
- **Format**: 32-bit integer (mode `I`)
- **Range**: ~13,000 – 65,535
- **Special value**: 65,535 = "infinite" distance (sky)
- **Normalization**: `depth / 65535.0` → [0, 1]

### Multi-View Generation

For JEPA training, each sample generates **6 views** with synchronized transforms:

| View Type | Count | Size | Scale Range | Purpose |
|-----------|-------|------|-------------|---------|
| Global | 2 | 224×224 | 0.4–1.0 | Context, defines "center" |
| Local | 4 | 96×96 | 0.05–0.4 | Fine details, must match center |

**Critical**: The same random crop is applied to both RGB and depth:

```python
# Get random crop parameters (shared)
i, j, h, w = RandomResizedCrop.get_params(img, scale, ratio)

# Apply to RGB (bilinear interpolation)
img_crop = resized_crop(img, i, j, h, w, size, BILINEAR)

# Apply to depth (NEAREST interpolation - preserves values!)
depth_crop = resized_crop(depth, i, j, h, w, size, NEAREST)
```

## Training

### Quick Start

```bash
# Full JEPA training
python train_depth_jepa.py \
    --epochs 50 \
    --bs 16 \
    --V_global 2 \
    --V_local 4 \
    --lamb 0.05 \
    --depth_weight 1.0 \
    --jepa_weight 0.5

# Or use SLURM
sbatch train_depth.sh
```

### Recommended Settings (RTX Pro 6000, 48GB)

| Model | Batch Size | Views | LR | Est. Time |
|-------|------------|-------|-------|-----------|
| vit_small | 16 | 2+4 | 1e-4 | ~1.5 hrs |
| vit_base | 8 | 2+4 | 5e-5 | ~2 hrs |

### Key Arguments

```
--V_global 2        # Number of global views (224×224)
--V_local 4         # Number of local views (96×96)
--lamb 0.05         # SIGReg weight in LeJEPA loss
--depth_weight 1.0  # Weight for depth supervision
--jepa_weight 0.5   # Weight for JEPA consistency loss
--neighborhoods 0,1,2,3  # Which data folders to use
```

## Evaluation Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| AbsRel | Mean absolute relative error | < 0.15 |
| RMSE | Root mean squared error | Lower is better |
| δ₁ | % pixels with max(pred/gt, gt/pred) < 1.25 | > 0.85 |
| δ₂ | % pixels with ratio < 1.25² | > 0.95 |
| δ₃ | % pixels with ratio < 1.25³ | > 0.98 |

## Model Details

### ViT Encoder

- **Base model**: `vit_small_patch16_224.augreg_in21k` (pretrained)
- **Feature dim**: 384 (vit_small) or 768 (vit_base)
- **Patch size**: 16×16
- **Dynamic size**: Enabled for multi-resolution views
- **Grad checkpointing**: Enabled for memory efficiency

### Depth Decoder

Progressive upsampling with transposed convolutions:

```
14×14 (patches) → 28 → 56 → 112 → 224×224 (depth)
```

Each block: `ConvTranspose2d + BatchNorm + GELU`

### Projection Head (for JEPA)

```
384 → 2048 → 512 (with BatchNorm, no affine on output)
```

## Files

```
aipi-540-cv-hackathon/
├── depth_ds.py          # Dataset with multi-view + symlink resolution
├── depth_model.py       # ViT encoder + decoder + projection head
├── train_depth_jepa.py  # JEPA training loop
├── train_depth.py       # Simple training (no multi-view)
├── train_depth.sh       # SLURM submission script
├── loss.py              # SIGReg, LeJEPA, VICReg losses
└── datalink/
    ├── neighbourhood/   # Symlinks to images/depths
    └── cache/           # HuggingFace blobs
```

## References

- **JEPA**: [A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf) (LeCun, 2022)
- **SIGReg**: Spectral regularization for self-supervised learning
- **Scale-Invariant Loss**: [Depth Map Prediction from a Single Image](https://arxiv.org/abs/1406.2283) (Eigen et al., 2014)
