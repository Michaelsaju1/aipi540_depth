
import torch
import sys
import os

def check_keys(checkpoint_path):
    print(f"Checking keys for: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print("Checkpoint not found!")
        return

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in ckpt:
            keys = list(ckpt["model_state_dict"].keys())
        else:
            keys = list(ckpt.keys())
        
        print("\nCheckpoint Keys (first 10):")
        for k in keys[:10]:
            print(k)

        print(f"\nTotal keys: {len(keys)}")
        
        # Check for common prefixes
        if all(k.startswith("module.") for k in keys):
            print("\nDetected 'module.' prefix (DataParallel/DDP).")
        elif all(k.startswith("encoder.") for k in keys):
             print("\nDetected 'encoder.' prefix.")

    except Exception as e:
        print(f"Error loading checkpoint: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_keys(sys.argv[1])
    else:
        print("Usage: python inspect_ckpt.py <path_to_checkpoint>")
