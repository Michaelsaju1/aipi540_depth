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

The project supports two main backbones selectable via `--model`:

1.  **Vision Transformer (ViT)**: `vit_small_patch16_224.augreg_in21k`
    *   Good for global context.
    *   Uses standard ViT encoder + simple upsampling decoder.
2.  **ResNet**: `resnet50.a1_in1k`
    *   **New**: Supports native high-resolution input ("full sized images") without patching.
    *   Good for preserving spatial details.
    *   Uses a 5-stage decoder (stride 32 -> 1) to recover full resolution.

```
RGB Image (B, 3, H, W)
        │
        ▼
┌─────────────────────────────┐
│   Backbone (ViT or ResNet)  │
└─────────────────────────────┘
        │
       ...
```

## Training

### Quick Start

```bash
# Standard JEPA training (ViT, crops)
python src/training/train_depth_jepa.py \
    --model vit_small_patch16_224.augreg_in21k \
    --bs 16 --V_global 2 --V_local 4

# Full Image Training (ResNet, adaptive size)
python src/training/train_depth_jepa.py \
    --model resnet50.a1_in1k \
    --full_size \
    --max_img_size 1500 \
    --bs 1 \
    --grad_accum 16
```

### Key Arguments

```
--model             # Model name (e.g. resnet50.a1_in1k, vit_small...)
--full_size         # Enable adaptive full-image training (no fixed crops)
--max_img_size      # Max dimension for adaptive resizing (default 1536)
--V_global/local    # Number of views (ignored in full_size mode)
```

## Testing & Evaluation

The testing pipeline (`src/test/test.py`) is robust to different input sizes and architectures.

*   **Patch-based Inference**: For ViT on large images, it automatically patches the image, runs inference, and reconstructs the full result.
*   **Adaptive Inference**: For ResNet, it can process full high-res images directly.
*   **Auto-Recovery**: Automatically detects checkpoint architecture mismatches (e.g., 4 vs 5 layer decoders) and reloads correctly.

```bash
python src/test/test.py \
    --model_path checkpoints/my_model.pt \
    --img_size 1024 \
    --save_dir test_results
```

## Dataset: DDOS (Depth from Driving Open Scenes)

**Source**: [benediktkol/DDOS](https://huggingface.co/datasets/benediktkol/DDOS)

### Data Structure

```
datalink/neighbourhood/
├── 0/
│   ├── image/     # RGB images (1280×720, PNG)
│   └── depth/     # Depth maps (1280×720, 32-bit int)
```

### Depth Format
*   **Resolution**: 1280 × 720
*   **Format**: 32-bit integer (mode `I`)
*   **Normalization**: `depth / 65535.0` → [0, 1]

## References

- **JEPA**: [A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf) (LeCun, 2022)
- **SIGReg**: Spectral regularization for self-supervised learning
- **Scale-Invariant Loss**: [Depth Map Prediction from a Single Image](https://arxiv.org/abs/1406.2283) (Eigen et al., 2014)
