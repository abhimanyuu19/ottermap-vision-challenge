from pathlib import Path
import random
import shutil

random.seed(42)

TILE_IMAGES = Path("data/tiles/images")
TILE_MASKS = Path("data/tiles/masks")

TRAIN_IMAGES = Path("data/train/images")
TRAIN_MASKS = Path("data/train/masks")

VAL_IMAGES = Path("data/val/images")
VAL_MASKS = Path("data/val/masks")

for folder in [TRAIN_IMAGES, TRAIN_MASKS, VAL_IMAGES, VAL_MASKS]:
    folder.mkdir(parents=True, exist_ok=True)

images = sorted(TILE_IMAGES.glob("*.png"))
random.shuffle(images)

split = int(len(images) * 0.8)

train_files = images[:split]
val_files = images[split:]

def copy_files(file_list, img_dest, mask_dest):
    for img in file_list:
        mask = TILE_MASKS / img.name

        shutil.copy(img, img_dest / img.name)
        shutil.copy(mask, mask_dest / mask.name)

copy_files(train_files, TRAIN_IMAGES, TRAIN_MASKS)
copy_files(val_files, VAL_IMAGES, VAL_MASKS)

print(f"Training images : {len(train_files)}")
print(f"Validation images : {len(val_files)}")