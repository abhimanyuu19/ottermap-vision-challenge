import rasterio
from pathlib import Path

IMAGE_DIR = Path("data/images")

for image_file in IMAGE_DIR.iterdir():
    print("=" * 50)
    print(f"Image: {image_file.name}")

    try:
        with rasterio.open(image_file) as src:
            print("Width :", src.width)
            print("Height:", src.height)
            print("CRS   :", src.crs)
            print("Transform:", src.transform)
    except Exception as e:
        print("Cannot open with Rasterio")
        print(e)