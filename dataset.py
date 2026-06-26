
"""
Dataset loader for turf semantic segmentation.

Loads image-mask pairs and prepares them for the SegFormer model.

Author: Abhimanyu Upadhyay
"""
from pathlib import Path

import numpy as np
from PIL import Image


from torch.utils.data import Dataset

from transformers import SegformerImageProcessor
from config import IMAGE_SIZE


class TurfDataset(Dataset):

    def __init__(self, image_dir, mask_dir):

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        self.images = sorted(self.image_dir.glob("*.png"))

        self.processor = SegformerImageProcessor(
    do_resize=True,
    size={
        "height": IMAGE_SIZE,
        "width": IMAGE_SIZE
    },
    do_normalize=True
)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_path = self.images[idx]

        mask_path = self.mask_dir / image_path.name

        image = Image.open(image_path).convert("RGB")
        
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        mask = Image.open(mask_path).convert("L")

        mask = np.array(mask)

        mask = (mask > 127).astype(np.int64)

        encoded = self.processor(
            images=image,
            segmentation_maps=mask,
            return_tensors="pt"
        )

        return {
            "pixel_values": encoded["pixel_values"].squeeze(),
            "labels": encoded["labels"].squeeze()
        }