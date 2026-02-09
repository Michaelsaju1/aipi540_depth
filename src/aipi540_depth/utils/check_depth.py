#!/usr/bin/env python3
"""Quick script to check depth image properties and value ranges."""

from PIL import Image
import numpy as np

depth_path = "datalink/neighbourhood/0/depth/1.png"
img_path = "datalink/neighbourhood/0/image/1.png"

# Load depth
depth = Image.open(depth_path)
depth_arr = np.array(depth)

print(f"Depth image path: {depth_path}")
print(f"Depth size: {depth.size}")
print(f"Depth mode: {depth.mode}")
print(f"Depth dtype: {depth_arr.dtype}")
print(f"Depth shape: {depth_arr.shape}")
print(f"Depth min: {depth_arr.min()}")
print(f"Depth max: {depth_arr.max()}")
print(f"Depth mean: {depth_arr.mean():.2f}")
print(f"Depth std: {depth_arr.std():.2f}")
print(f"Unique values (first 20): {np.unique(depth_arr)[:20]}")
print(f"Total unique values: {len(np.unique(depth_arr))}")

# Also check the RGB image
print("\n--- RGB Image ---")
img = Image.open(img_path)
img_arr = np.array(img)
print(f"Image size: {img.size}")
print(f"Image mode: {img.mode}")
print(f"Image shape: {img_arr.shape}")
