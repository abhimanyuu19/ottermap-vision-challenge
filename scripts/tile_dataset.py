from pathlib import Path
from PIL import Image

TILE_SIZE = 512

IMAGE_DIR = Path("data/images")
MASK_DIR = Path("data/masks")

OUT_IMG = Path("data/tiles/images")
OUT_MASK = Path("data/tiles/masks")

OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_MASK.mkdir(parents=True, exist_ok=True)

image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]

for image_path in IMAGE_DIR.iterdir():

    if image_path.suffix.lower() not in image_extensions:
        continue

    stem = image_path.stem

    mask_path = MASK_DIR / f"{stem}.png"

    if not mask_path.exists():
        print(f"Mask missing for {stem}")
        continue

    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    width, height = image.size

    tile_id = 0

    for y in range(0, height, TILE_SIZE):
        for x in range(0, width, TILE_SIZE):

            img_tile = image.crop((x, y, x + TILE_SIZE, y + TILE_SIZE))
            mask_tile = mask.crop((x, y, x + TILE_SIZE, y + TILE_SIZE))

            img_tile.save(
                OUT_IMG / f"{stem}_{tile_id}.png"
            )

            mask_tile.save(
                OUT_MASK / f"{stem}_{tile_id}.png"
            )

            tile_id += 1

    print(f"{stem}: {tile_id} tiles created")

print("Done!")