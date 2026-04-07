import os
import math
from PIL import Image

# --------------------------------------------------
# SETTINGS  (edit only this part)
# --------------------------------------------------

background_path = "background_samples/imagenet-indoor.JPEG"

objects = [
    {
        "name": "vase",
        "path": "foreground_samples/vase/vase1.png",
        "area": 0.05,        # % of background area
        "coord": (0.25, 0.82)
    },
    {
        "name": "toaster",
        "path": "foreground_samples/toaster/toaster1.png",
        "area": 0.03,
        "coord": (0.75, 0.80)
    }
]

output_dir = "vase-toaster-indoor"
bg_resolution = (250, 250)
rotations = range(360)

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def resize_by_area(fg, bg, area_ratio):
    bg_area = bg.width * bg.height
    target_area = bg_area * area_ratio

    fg_area = fg.width * fg.height
    scale = math.sqrt(target_area / fg_area)

    new_size = (int(fg.width * scale), int(fg.height * scale))
    return fg.resize(new_size, Image.LANCZOS)


def paste_center(bg, fg, coord):
    cx = int(coord[0] * bg.width)
    cy = int(coord[1] * bg.height)

    x = cx - fg.width // 2
    y = cy - fg.height // 2

    bg.paste(fg, (x, y), fg)
    return bg


# --------------------------------------------------
# GENERATOR
# --------------------------------------------------

os.makedirs(output_dir, exist_ok=True)

bg_original = Image.open(background_path).convert("RGBA")
bg_original = bg_original.resize(bg_resolution)

fg_images = []
for obj in objects:
    img = Image.open(obj["path"]).convert("RGBA")
    fg_images.append(img)

print("Generating images...")

for angle in rotations:
    bg = bg_original.copy()

    for obj, fg_img in zip(objects, fg_images):

        fg = resize_by_area(fg_img, bg, obj["area"])
        fg = fg.rotate(angle, expand=True)

        bg = paste_center(bg, fg, obj["coord"])

    bg.convert("RGB").save(f"{output_dir}/{output_dir}_{angle}.png", quality=95)

print("Done ✓")
