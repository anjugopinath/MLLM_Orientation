# import os

# input_foreground_path = "/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/data/filtered/composition/foreground"
# foreground_img_list = ["n01629819_74.JPEG","n02108422_4102.JPEG","n04310018_1941.JPEG"]
# foreground_img_names = ["lizard","dog","train"]

# input_background_path = "/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/data/filtered/composition/background"
# background_img_list = ["n01443537_223.JPEG","n03761084_821.JPEG","n03888257_482.JPEG"]
# background_img_names = ["fish","indoor","beach"]

# output_path = "/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/data/filtered/natural/imagenet/rotated_cropped"

# for index,img in enumerate(foreground_img_list):
#     foreground_img_path = os.path.join(input_foreground_path,img)

import os
from PIL import Image

input_foreground_path = "/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/data/filtered/composition/foreground"
foreground_img_list = ["n01629819_74.JPEG","n02108422_4102.JPEG","n04310018_1941.JPEG"]
foreground_img_names = ["lizard","dog","train"]

input_background_path = "/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/data/filtered/composition/background"
background_img_list = ["n01443537_223.JPEG","n03761084_821.JPEG","n03888257_482.JPEG"]
background_img_names = ["fish","indoor","beach"]


output_path = "/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/data/filtered/natural/imagenet/rotated_cropped"



for img_file, name in zip(foreground_img_list, foreground_img_names):

    if(name=="train"):
        CROP_SIZE = 200
    else:
        CROP_SIZE = 250

    img_path = os.path.join(input_foreground_path, img_file)
    img = Image.open(img_path).convert("RGB")
    save_dir = os.path.join(output_path, name, "1_degree")
    os.makedirs(save_dir, exist_ok=True)

    for angle in range(180):  # 0 → 179

        # rotate (expand so nothing gets clipped)
        rotated = img.rotate(-angle)

        w, h = rotated.size

        # center crop 300x300
        left = (w - CROP_SIZE) // 2
        top = (h - CROP_SIZE) // 2
        right = left + CROP_SIZE
        bottom = top + CROP_SIZE

        cropped = rotated.crop((left, top, right, bottom))    

        img_save_name = f"{name}_{str(angle)}.png"
        save_path = os.path.join(save_dir, img_save_name)
        cropped.save(save_path)

for img_file, name in zip(background_img_list, background_img_names):

    if(name in ["fish","indoor"]):
        CROP_SIZE = 200
    else:
        CROP_SIZE = 250

    img_path = os.path.join(input_background_path, img_file)
    img = Image.open(img_path).convert("RGB")
    save_dir = os.path.join(output_path, name, "1_degree")
    os.makedirs(save_dir, exist_ok=True)

    for angle in range(180):  # 0 → 179

        # rotate (expand so nothing gets clipped)
        rotated = img.rotate(-angle)

        w, h = rotated.size

        # center crop 300x300
        left = (w - CROP_SIZE) // 2
        top = (h - CROP_SIZE) // 2
        right = left + CROP_SIZE
        bottom = top + CROP_SIZE

        cropped = rotated.crop((left, top, right, bottom))    

        img_save_name = f"{name}_{str(angle)}.png"
        save_path = os.path.join(save_dir, img_save_name)
        cropped.save(save_path)

print("Done ✓")
