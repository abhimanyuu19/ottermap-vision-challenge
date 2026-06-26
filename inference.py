"""
Inference script for SegFormer semantic segmentation.

This script:
- Loads a trained SegFormer model.
- Predicts a segmentation mask for a single image.
- Saves the predicted mask.
- Creates an overlay visualization.

Author: Abhimanyu Upadhyay
"""
import argparse

import numpy as np
import torch
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

from config import *

# -----------------------------
# Load model
# -----------------------------
processor = SegformerImageProcessor(
    do_resize=True,
    size={"height": IMAGE_SIZE, "width": IMAGE_SIZE},
    do_normalize=True,
)

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True,
)

from pathlib import Path

weights_path = WEIGHTS_DIR / "best_model.pth"

if weights_path.exists():
    model.load_state_dict(
        torch.load(weights_path, map_location=DEVICE)
    )
    print("[OK] Loaded trained model.")
else:
    print("=" * 60)
    print("WARNING: No trained model found.")
    print(f"Expected location: {weights_path}")
    print("Using pretrained SegFormer model.")
    print("Predictions will not represent the fine-tuned model.")
    print("=" * 60)

model.to(DEVICE)
model.eval()

# -----------------------------
# Arguments
# -----------------------------
parser = argparse.ArgumentParser()

parser.add_argument(
    "--image",
    required=True,
    help="Path to input image"
)

args = parser.parse_args()

# -----------------------------
# Load image
# -----------------------------
image = Image.open(args.image).convert("RGB")

original_size = image.size

inputs = processor(
    images=image,
    return_tensors="pt"
)

pixel_values = inputs["pixel_values"].to(DEVICE)

# -----------------------------
# Prediction
# -----------------------------
with torch.no_grad():

    outputs = model(pixel_values=pixel_values)

prediction = outputs.logits.argmax(dim=1)[0].cpu().numpy()

# -----------------------------
# Save mask
# -----------------------------
mask = (prediction * 255).astype(np.uint8)

mask_image = Image.fromarray(mask)

mask_image = mask_image.resize(original_size)

OUTPUT_DIR.mkdir(exist_ok=True)

image_name = Path(args.image).stem

mask_path = OUTPUT_DIR / f"{image_name}_mask.png"

mask_image.save(mask_path)

print("=" * 60)
print(f"Mask saved successfully:")
print(mask_path)

# -----------------------------
# Overlay
# -----------------------------
overlay = np.array(image)

mask_np = np.array(mask_image)

overlay[mask_np > 0] = [255, 0, 0]

overlay = Image.fromarray(overlay)

overlay_path = OUTPUT_DIR / f"{image_name}_overlay.png"

overlay.save(overlay_path)

print(f"Overlay saved successfully:")
print(overlay_path)
print("=" * 60)