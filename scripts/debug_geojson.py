import geopandas as gpd
from pathlib import Path
from PIL import Image

gdf = gpd.read_file("data/labels/1.geojson")

img = Image.open("data/images/1.jpg")

width, height = img.size

print("=" * 50)
print("Image Size")
print("=" * 50)
print("Width :", width)
print("Height:", height)

print("\n")

print("=" * 50)
print("GeoJSON Bounds")
print("=" * 50)

print(gdf.total_bounds)

print("\n")

print("=" * 50)
print("Geometry Types")
print("=" * 50)

print(gdf.geom_type.value_counts())

print("\n")

print("=" * 50)
print("Number of Features")
print("=" * 50)

print(len(gdf))

print("\n")

print("=" * 50)
print("First Geometry")
print("=" * 50)

print(gdf.geometry.iloc[0])