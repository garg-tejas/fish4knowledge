import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import logging
import re

class FishSegmentationClassificationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # Check if root directory exists
        if not os.path.exists(root_dir):
            print(f"[ERROR] Root directory does not exist: {root_dir}")
            return
            
        # Look for all class folders
        try:
            class_folders = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
            print(f"[DEBUG] Found {len(class_folders)} class folders in {root_dir}: {class_folders}")
        except Exception as e:
            print(f"[ERROR] Could not list directories in {root_dir}: {e}")
            return
        
        for class_folder in class_folders:
            class_folder_path = os.path.join(root_dir, class_folder)
            image_folder_path = os.path.join(class_folder_path, "images")
            mask_folder_path = os.path.join(class_folder_path, "masks")
            
            if not os.path.exists(image_folder_path):
                print(f"[WARNING] Images folder not found: {image_folder_path}")
                continue
                
            if not os.path.exists(mask_folder_path):
                print(f"[WARNING] Masks folder not found: {mask_folder_path}")
                continue
                
            # Match all image types: .png, .jpg, .jpeg
            fish_images = []
            for ext in ('*.png', '*.jpg', '*.jpeg'):
                fish_images.extend(glob.glob(os.path.join(image_folder_path, ext)))
            
            if not fish_images:
                print(f"[WARNING] No images found in {image_folder_path}")
                continue
                
            print(f"[DEBUG] Found {len(fish_images)} images in {image_folder_path}")
            
            # Get all mask files for faster lookup
            mask_files = []
            for ext in ('*.png', '*.jpg', '*.jpeg'):
                mask_files.extend(glob.glob(os.path.join(mask_folder_path, ext)))
            mask_filenames = [os.path.basename(f) for f in mask_files]
            
            for fish_img in fish_images:
                base_name = os.path.basename(fish_img)
                
                # Try different mask naming patterns
                mask_found = False
                
                # 1. Try exact match
                if base_name in mask_filenames:
                    mask_path = os.path.join(mask_folder_path, base_name)
                    mask_found = True
                
                # 2. Try replacing "fish_" with "mask_" in the filename
                elif base_name.startswith("fish_"):
                    mask_name = "mask_" + base_name[5:]
                    if mask_name in mask_filenames:
                        mask_path = os.path.join(mask_folder_path, mask_name)
                        mask_found = True
                
                # 3. Try just the numeric part (remove fish_ prefix)
                elif base_name.startswith("fish_"):
                    mask_name = base_name[5:]
                    if mask_name in mask_filenames:
                        mask_path = os.path.join(mask_folder_path, mask_name)
                        mask_found = True
                
                # 4. Extract ID numbers and look for matching mask with same IDs
                if not mask_found:
                    # Extract the numeric parts from the filename
                    number_pattern = re.compile(r'(\d+)')
                    numbers = number_pattern.findall(base_name)
                    
                    if numbers and len(numbers) >= 2:
                        # Look for masks with the same ID numbers
                        potential_masks = [mf for mf in mask_filenames if all(num in mf for num in numbers)]
                        if potential_masks:
                            mask_path = os.path.join(mask_folder_path, potential_masks[0])
                            mask_found = True
                            print(f"[INFO] Found matching mask for {base_name} -> {potential_masks[0]}")
                
                if mask_found:
                    try:
                        label = int(class_folder) - 1  # For 01, 02, etc. folders
                    except ValueError:
                        label = class_folders.index(class_folder)
                        
                    self.samples.append((fish_img, mask_path, label))
                else:
                    print(f"[WARNING] Mask not found for image: {fish_img}")
        
        print(f"[INFO] Loaded {len(self.samples)} samples from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # L = grayscale
        
        if self.transform:
            # Apply same resize to both image and mask
            image = self.transform(image)
            
            # Create a separate transform for mask that matches image resize but preserves mask values
            mask_transform = transforms.Compose([
                transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor()
            ])
            mask = mask_transform(mask)
        else:
            # Default conversion to tensors with resize
            image_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])
            mask_transform = transforms.Compose([
                transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor()
            ])
            image = image_transform(image)
            mask = mask_transform(mask)
        
        return image, mask, label