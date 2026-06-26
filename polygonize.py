"""
Convert predicted segmentation masks into GeoJSON polygons.

This script:
- Reads predicted mask images.
- Extracts polygon boundaries using OpenCV.
- Converts contours to Shapely polygons.
- Saves the results as GeoJSON files.

Author: Abhimanyu Upadhyay
"""
from pathlib import Path
import cv2
import geopandas as gpd
from shapely.geometry import Polygon

OUTPUT_DIR = Path("outputs")
RESULT_DIR = Path("results")

RESULT_DIR.mkdir(exist_ok=True)

mask_files = list(OUTPUT_DIR.glob("*_mask.png"))

if len(mask_files) == 0:
    print("No predicted masks found.")
    exit()

for mask_file in mask_files:

    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    polygons = []

    for contour in contours:

        if len(contour) < 3:
            continue

        contour = contour.squeeze()

        if contour.ndim != 2:
            continue

        poly = Polygon(contour)

        if poly.is_valid:
            polygons.append(poly)
            
            gdf = gpd.GeoDataFrame(
    {
        "id": range(1, len(polygons) + 1),
        "source_mask": [mask_file.name] * len(polygons),
    },
    geometry=polygons,
)
            output_file = RESULT_DIR / f"{mask_file.stem}.geojson"
            gdf.to_file(output_file, driver="GeoJSON")
            print("=" * 60)
print("GeoJSON saved successfully:")
print(output_file)
print(f"Polygons extracted: {len(polygons)}")
print("=" * 60)
print("\nFinished.")