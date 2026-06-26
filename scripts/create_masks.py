import geopandas as gpd
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

IMAGE_DIR = Path("data/images")
LABEL_DIR = Path("data/labels")
MASK_DIR = Path("data/masks")

MASK_DIR.mkdir(exist_ok=True)

for geojson in LABEL_DIR.glob("*.geojson"):

    name = geojson.stem

    image_path = None

    for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
        p = IMAGE_DIR / f"{name}{ext}"
        if p.exists():
            image_path = p
            break

    if image_path is None:
        continue

    image = Image.open(image_path)

    width, height = image.size

    gdf = gpd.read_file(geojson)

    minx, miny, maxx, maxy = gdf.total_bounds

    mask = np.zeros((height, width), dtype=np.uint8)

    for geom in gdf.geometry:

        if geom is None:
            continue

        polygons = []

        if geom.geom_type == "Polygon":
            polygons = [geom]

        elif geom.geom_type == "MultiPolygon":
            polygons = list(geom.geoms)

        for poly in polygons:

            coords = np.array(poly.exterior.coords)

            xs = ((coords[:,0]-minx)/(maxx-minx))*width
            ys = ((maxy-coords[:,1])/(maxy-miny))*height

            pts = np.stack([xs,ys],axis=1).astype(np.int32)

            cv2.fillPoly(mask,[pts],255)

    cv2.imwrite(str(MASK_DIR/f"{name}.png"),mask)

    print(f"{name} done")

print("Finished")