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
    def __init__(self, split, V_global=2, V_local=4, device="cuda", global_img_size=224, local_img_size=96, dataset="inet100"):
        self.V_global = V_global
        self.V_local = V_local
        self.split = split
        self.global_img_size = global_img_size
        self.local_img_size = local_img_size
        
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
        if dataset == "cifar10":
            self.ds = load_dataset("cifar10", split=self.split)
        elif dataset == "inet100":
            self.inet_dir = "/home/users/aho13/jepa_tests/data/cache/datasets--clane9--imagenet-100/snapshots/0519dc2f402a3a18c6e57f7913db059215eee25b/data/"
            filenames = {
                "train": self.inet_dir + "train-*.parquet",
                "val": self.inet_dir + "validation*.parquet",
            }
            self.ds = load_dataset("parquet", data_files=filenames, split=self.split)
        
        # --- CUSTOM LOCAL DATASET LOGIC ---
        elif dataset == "local_13":
            # This combines the Current Working Directory with your image path
            # Result: /home/users/aho13/aipi-540-cv-hackathon/datalink/neighbourhood/13/image
            img_dir = os.path.join(os.getcwd(), "datalink", "neighbourhood", "13", "image")
            
            # Find all png files
            all_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
            
            if not all_files:
                raise ValueError(f"No .png images found in {img_dir}")

            # Shuffle deterministically (using fixed seed so train/test don't leak)
            rng = random.Random(42)
            rng.shuffle(all_files)

            # 80% / 20% Split
            split_idx = int(0.8 * len(all_files))
            
            if self.split == 'train':
                file_list = all_files[:split_idx]
            else:
                file_list = all_files[split_idx:]

            # Internal wrapper to make list behave like the HF dataset
            class LocalWrapper:
                def __init__(self, files):
                    self.files = files
                def __len__(self):
                    return len(self.files)
                def __getitem__(self, i):
                    return {
                        "image": Image.open(self.files[i]).convert("RGB"),
                        "label": 13 
                    }
            
            self.ds = LocalWrapper(file_list)
        
        else:
            raise ValueError(f"Dataset {dataset} not supported")

    def _load_image(self, entry):
        """Helper to handle safe image extraction from row entry."""
        if "image" in entry:
            return entry["image"] # PIL Image already returned by LocalWrapper
        elif "img" in entry:
            return entry["img"].convert("RGB")
        else:
            raise ValueError("Image not found in entry")

    def __getitem__(self, i):
        entry = self.ds[i]
        # Direct access because LocalWrapper returns PIL image in "image" key
        img = entry["image"] 
        label = entry["label"]
        
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
        return len(self.ds)

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    print(f"Current Working Directory: {os.getcwd()}")
    
    # 1. Initialize Train (First 80%)
    print("--- Loading Train Split ---")
    train_ds = HFDataset(split="train", dataset="local_13")
    print(f"Train Dataset Size: {len(train_ds)}")

    # 2. Initialize Test (Last 20%)
    print("--- Loading Test Split ---")
    test_ds = HFDataset(split="test", dataset="local_13")
    print(f"Test Dataset Size: {len(test_ds)}")

    # 3. Test loading one image
    if len(train_ds) > 0:
        views, label = train_ds[0]
        print(f"Success! Loaded one training sample. Label: {label}")
        print(f"Views generated: {len(views)}")
    else:
        print("Dataset is empty. Check your paths.")