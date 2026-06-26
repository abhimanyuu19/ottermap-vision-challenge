from PIL import Image

image = Image.open("data/images/1.jpg").convert("RGB")
mask = Image.open("data/masks/1.png").convert("L")

mask = mask.resize(image.size)

overlay = image.copy()

pixels = overlay.load()
mask_pixels = mask.load()

for y in range(image.height):
    for x in range(image.width):
        if mask_pixels[x, y] > 0:
            r, g, b = pixels[x, y]
            pixels[x, y] = (255, 0, 0)

overlay.save("outputs/overlay_1.png")

print("Overlay saved!")