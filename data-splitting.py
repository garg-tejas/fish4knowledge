import os
import shutil
import random
from tqdm import tqdm

# ==== CONFIGURATION ====
source_dir = "data/FishDataset"
output_dir = "data"
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

random.seed(42)

# ==== FUNCTIONS ====

def make_dirs(base_dir, split_name):
    """Create the destination directory structure"""
    split_dir = os.path.join(base_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    return split_dir

def copy_file(src_path, dst_path):
    """Copy a file from src to dst."""
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy2(src_path, dst_path)

make_dirs(output_dir, "train")
make_dirs(output_dir, "val")
make_dirs(output_dir, "test")

if not os.path.exists(source_dir):
    os.makedirs(source_dir, exist_ok=True)
    print(f"Created empty source directory: {source_dir}")
    print("Please place your Fish4Knowledge dataset in this directory and run again.")
    exit(1)

fish_folders = [f for f in os.listdir(source_dir) if f.startswith("fish_")]

if not fish_folders:
    print(f"No fish folders found in {source_dir}!")
    print("Expected folder structure: data/FishDataset/fish_1, data/FishDataset/mask_1, etc.")
    exit(1)

print(f"Found {len(fish_folders)} fish classes")

for fish_folder in tqdm(fish_folders, desc="Processing Fish Folders"):
    tracking_id = fish_folder.split("_")[1]
    fish_path = os.path.join(source_dir, fish_folder)
    mask_path = os.path.join(source_dir, f"mask_{tracking_id}")
    
    if not os.path.exists(mask_path):
        print(f"Warning: Missing mask folder for {fish_folder}. Skipping.")
        continue
    
    try:
        image_files = sorted(os.listdir(fish_path))
        mask_files = sorted(os.listdir(mask_path))
    except:
        print(f"Error reading files from {fish_path} or {mask_path}. Skipping.")
        continue
    
    paired_files = []
    for img_file in image_files:
        if not img_file.endswith('.png'):
            continue
            
        parts = img_file.split("_", 2)
        if len(parts) < 3:
            continue
            
        tracking = parts[1]
        unique = parts[2].replace(".png", "")
        
        mask_file = f"mask_{tracking}_{unique}.png"
        if mask_file in mask_files:
            paired_files.append((img_file, mask_file))
    
    if not paired_files:
        print(f"No valid image-mask pairs found for {fish_folder}. Skipping.")
        continue
        
    random.shuffle(paired_files)
    total = len(paired_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_pairs = paired_files[:train_end]
    val_pairs = paired_files[train_end:val_end]
    test_pairs = paired_files[val_end:]
    
    splits = {
        "train": train_pairs,
        "val": val_pairs,
        "test": test_pairs
    }
    
    for split_name, pairs in splits.items():
        os.makedirs(os.path.join(output_dir, split_name, tracking_id, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split_name, tracking_id, "masks"), exist_ok=True)
        
        for img_file, mask_file in pairs:
            img_src = os.path.join(fish_path, img_file)
            mask_src = os.path.join(mask_path, mask_file)
            
            img_dst = os.path.join(output_dir, split_name, tracking_id, "images", img_file)
            mask_dst = os.path.join(output_dir, split_name, tracking_id, "masks", mask_file)
            
            copy_file(img_src, img_dst)
            copy_file(mask_src, mask_dst)

print("\n✅ Dataset splitting completed successfully!")
print(f"Check {output_dir}/train, {output_dir}/val, and {output_dir}/test for the organized dataset.")