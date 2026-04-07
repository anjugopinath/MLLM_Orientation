# import os
# import cv2
# import numpy as np
# from PIL import Image

# # ------------------------------------------------------------
# # UTILITY FUNCTIONS
# # ------------------------------------------------------------
# def pad_to_square(image):
#     """Pads an image to a square shape using black borders."""
#     h, w = image.shape[:2]
#     max_dim = max(h, w)
    
#     # Calculate padding amounts
#     pad_h = max_dim - h
#     pad_w = max_dim - w
    
#     # Pad top/bottom, left/right equally
#     top = pad_h // 2
#     bottom = pad_h - top
#     left = pad_w // 2
#     right = pad_w - left
    
#     # Pad the image
#     padded_image = cv2.copyMakeBorder(
#         image, top, bottom, left, right, 
#         cv2.BORDER_CONSTANT, value=[0, 0, 0] # Black border
#     )
    
#     return padded_image

# def generate_synthetic_background(type_name, size=(375, 500), line_thickness=3, cell_size=50):
#     """Generate a synthetic background based on the given type."""
#     h, w = size
#     bg = np.ones((h, w, 3), dtype=np.uint8) * 255  # white base

#     if type_name == "chessboard":
#         for y in range(0, h, cell_size):
#             for x in range(0, w, cell_size):
#                 if ((x // cell_size) + (y // cell_size)) % 2 == 0:
#                     bg[y:y+cell_size, x:x+cell_size] = 0  # black

#     elif type_name == "grid_lines":
#         for y in range(0, h, cell_size):
#             cv2.line(bg, (0, y), (w, y), (0, 0, 0), line_thickness)
#         for x in range(0, w, cell_size):
#             cv2.line(bg, (x, 0), (x, h), (0, 0, 0), line_thickness)

#     elif type_name == "horizontal_lines":
#         for y in range(0, h, cell_size):
#             cv2.line(bg, (0, y), (w, y), (0, 0, 0), line_thickness)

#     elif type_name == "vertical_lines":
#         for x in range(0, w, cell_size):
#             cv2.line(bg, (x, 0), (x, h), (0, 0, 0), line_thickness)

#     else:
#         # Fallback to a solid color if type is unknown
#         bg[:] = (200, 200, 200)

#     return bg

# # ------------------------------------------------------------
# # MAIN COMPOSITING FUNCTION (Modified)
# # ------------------------------------------------------------
# def extract_and_superimpose_circle_all_rotations_counterrot(
#     source_path, shape, target_path, radius, output_dir, rot_option, output_filename
# ):
#     print("--- Starting Counter-Rotation Process ---")
#     output_base_name = output_filename.split('/')[-1].split('.')[0]
#     output_subdir_name = output_base_name + "_" + rot_option

#     # Load images
#     src_orig = cv2.imread(source_path, cv2.IMREAD_COLOR)
#     tgt_orig = cv2.imread(target_path, cv2.IMREAD_COLOR)
    
#     # --- CORE CHANGE: Pad source image to be square for perfect circle mask ---
#     src = pad_to_square(src_orig)
#     print(f"Padded Source image shape (H, W, C): {src.shape}")
#     # -------------------------------------------------------------------------

#     src_center = (src.shape[1] // 2, src.shape[0] // 2)
#     tgt_center = (tgt_orig.shape[1] // 2, tgt_orig.shape[0] // 2)

#     # --- Gradual alpha mask for smooth blending ---
#     mask = np.zeros(src.shape[:2], dtype=np.float32)
#     cv2.circle(mask, src_center, radius, 1.0, -1) 

#     inner_radius_factor = 0.8
#     feather_width = radius * (1 - inner_radius_factor)
#     inner_radius = int(radius * inner_radius_factor)

#     # --- ADDED PRINT: Opaque foreground diameter (based on inner_radius) ---
#     ORIGINAL_OPAQUE_DIAMETER = 2 * inner_radius
#     print(f"Original (Unscaled) Opaque Foreground Diameter: {ORIGINAL_OPAQUE_DIAMETER} pixels")
#     # ------------------------------------------------------------------------

#     for y in range(src.shape[0]):
#         for x in range(src.shape[1]):
#             dist = np.sqrt((x - src_center[0])**2 + (y - src_center[1])**2)
#             if dist > inner_radius:
#                 alpha_val = 1.0 - (dist - inner_radius) / feather_width
#                 mask[y, x] = max(0.0, alpha_val)

#     src_float = src.astype(np.float32) / 255.0
#     mask_3ch = cv2.merge([mask] * 3)
#     src_circle = (src_float * mask_3ch * 255).astype(np.uint8)

#     # Crop with padding (using padded 'src' image)
#     x, y = src_center
#     crop_padding = int(radius * (1 - inner_radius_factor) * 1.5)
#     x1, y1 = max(0, x - radius - crop_padding), max(0, y - radius - crop_padding)
#     x2, y2 = min(src.shape[1], x + radius + crop_padding), min(src.shape[0], y + radius + crop_padding)
#     cropped_circle = src_circle[y1:y2, x1:x2]
#     cropped_mask = mask[y1:y2, x1:x2]
#     orig_patch_size = cropped_circle.shape[0] # The dimension of the cropped square patch

#     tgt_h, tgt_w = tgt_orig.shape[:2]
#     # --- ADDED PRINT: Background image size ---
#     print(f"Background image size (H, W): {tgt_h}, {tgt_w}")
#     # ------------------------------------------

#     for scale_idx in range(60):
#         if scale_idx not in (0, 3, 15):
#             continue

#         current_scale_factor = 1.0 / (scale_idx + 1)
#         max_possible_radius_target = min(tgt_w, tgt_h) / 2
        
#         # Calculate new patch size
#         new_dim = int(orig_patch_size * current_scale_factor)
#         if new_dim % 2 != 0:
#             new_dim += 1
#         size = (new_dim, new_dim)

#         if new_dim <= 0 or new_dim > min(tgt_w, tgt_h):
#             print(f"Skipping scale {scale_idx + 1}: Resized patch size {new_dim}x{new_dim} is invalid or too large.")
#             continue
            
#         # --- ADDED PRINT: Rescaled opaque diameter ---
#         scaled_opaque_diameter = int(ORIGINAL_OPAQUE_DIAMETER * current_scale_factor)
#         print(f"\n--- Scale Factor 1/{scale_idx + 1} ({current_scale_factor:.4f}) ---")
#         print(f"Rescaled Opaque Foreground Diameter: {scaled_opaque_diameter} pixels")
#         print(f"Resized Patch Dimensions (including padding): {new_dim}x{new_dim}")
#         # ---------------------------------------------

#         interpolation = cv2.INTER_AREA if current_scale_factor < 1.0 else cv2.INTER_LINEAR
#         circle_resized = cv2.resize(cropped_circle, size, interpolation=interpolation)
#         mask_resized = cv2.resize(cropped_mask, size, interpolation=cv2.INTER_LINEAR)

#         subfolder = os.path.join(output_dir, output_subdir_name, f"scale{scale_idx + 1}")
#         os.makedirs(subfolder, exist_ok=True)

#         for init_angle in range(0, -360, -1):
#             angle = init_angle
            
#             # Convert resized patch and mask to PIL images for potential rotation
#             pil_circle = Image.fromarray(cv2.cvtColor(circle_resized, cv2.COLOR_BGR2RGB))
#             pil_mask = Image.fromarray((mask_resized * 255).astype(np.uint8))

#             # --- FOREGROUND ROTATION CONTROL BLOCK ---
            
#             # UNCOMMENT TO ROTATE FOREGROUND
#             if(rot_option == "b.5Rot_f.5Rot_1degreeTotal"):
#                 rotated_circle = pil_circle.rotate(angle, resample=Image.BICUBIC, expand=False)
#                 rotated_mask = pil_mask.rotate(angle, resample=Image.BICUBIC, expand=False)

#                 circle_np = cv2.cvtColor(np.array(rotated_circle), cv2.COLOR_RGB2BGR)
#                 mask_np = np.array(rotated_mask).astype(np.float32) / 255.0
#             # ---------- END ----------

#             # UNCOMMENT TO KEEP FOREGROUND STATIC
#             if(rot_option == "bRot_fStat"):
#                 circle_np = cv2.cvtColor(np.array(pil_circle), cv2.COLOR_RGB2BGR)
#                 mask_np = np.array(pil_mask).astype(np.float32) / 255.0
#             # ---------- END ----------
            
#             # -------------------------------------------------------------

#             mask_3ch_rotated = cv2.merge([mask_np] * 3)

#             # --- Background counter-rotation (opposite direction) ---
#             opposite_angle = -angle
#             pil_bg = Image.fromarray(cv2.cvtColor(tgt_orig, cv2.COLOR_BGR2RGB))

#             # Rotate background with expand=True using LANCZOS for higher quality resampling
#             rotated_bg = pil_bg.rotate(opposite_angle, resample=3, expand=True) 
#             rot_w, rot_h = rotated_bg.size

#             # Crop center to original target size
#             cx, cy = rot_w // 2, rot_h // 2
#             left = cx - tgt_w // 2
#             upper = cy - tgt_h // 2
#             right = left + tgt_w
#             lower = upper + tgt_h
#             rotated_bg_cropped = rotated_bg.crop((left, upper, right, lower))

#             # Convert to BGR for OpenCV and smooth edges
#             tgt = cv2.cvtColor(np.array(rotated_bg_cropped), cv2.COLOR_RGB2BGR)
#             # Apply stronger Gaussian blur to smooth serrated edges
#             tgt = cv2.GaussianBlur(tgt, (3, 3), sigmaX=1.0, sigmaY=1.0) 

#             # Paste foreground at exact target center
#             tx, ty = tgt_center
#             rot_h, rot_w = circle_np.shape[:2]
#             tx1, ty1 = tx - rot_w // 2, ty - rot_h // 2
#             tx2, ty2 = tx1 + rot_w, ty1 + rot_h

#             roi = tgt[ty1:ty2, tx1:tx2].astype(np.float32) / 255.0
#             blended = circle_np.astype(np.float32) / 255.0 * mask_3ch_rotated + roi * (1 - mask_3ch_rotated)
#             tgt[ty1:ty2, tx1:tx2] = (blended * 255).astype(np.uint8)

#             out_path = os.path.join(subfolder, f"{output_base_name}_{abs(init_angle)}.png")
#             cv2.imwrite(out_path, tgt)

#         print(f"Finished scale {scale_idx} for {output_base_name}: {abs(init_angle)} total rotations completed.")
#     print("********************************************")


# # ------------------------------------------------------------
# # DRIVER CODE 
# # ------------------------------------------------------------
# foreground_list = ['dog', 'lizard', 'train', 'rectangleTriangle']
# background_list = ["chessboard", "grid_lines", "horizontal_lines", "vertical_lines"]
# # background_list =  synthetic_backgrounds
# synthetic_backgrounds = ["chessboard", "grid_lines", "horizontal_lines", "vertical_lines"]
# rot_option = "bRot_fStat" # b.5Rot_f.5Rot_1degreeTotal, bRot_fStat
# # Directory for synthetic backgrounds
# synthetic_bg_dir = "/<dirname>/<name>/g-t_embedding_classifier/data/filtered/composition/synthetic_backgrounds"
# os.makedirs(synthetic_bg_dir, exist_ok=True)

# # Generate and save synthetic backgrounds once
# for bg_type in synthetic_backgrounds:
#     bg_path = os.path.join(synthetic_bg_dir, f"{bg_type}.png")
#     if not os.path.exists(bg_path):
#         print(f"Generating synthetic background: {bg_type}")
#         # Using a standard small size for synthetic bgs as they are patterns
#         bg = generate_synthetic_background(bg_type, size=(375, 500)) 
#         cv2.imwrite(bg_path, bg)

# for foreground in foreground_list:
#     for background in background_list:
#         combination = f"{foreground}_on_{background}"
        
#         # Process only 'dog' combinations as suggested by the code's pattern
#         if "dog" not in combination:
#             continue

#         print(f"\n========================================================")
#         print(f"Processing: {foreground} on {background} (Counter-Rotation, Static Foreground)")
#         print(f"========================================================")

#         # --- Foreground paths ---
#         if foreground == 'dog':
#             source_path = "/<dirname>/<name>/g-t_embedding_classifier/data/filtered/composition/foreground/n02108422_4102.JPEG"
#         elif foreground == 'lizard':
#             source_path = "/<dirname>/<name>/g-t_embedding_classifier/data/filtered/composition/foreground/n01629819_74.JPEG"
#         elif foreground == 'train':
#             source_path = "/<dirname>/<name>/g-t_embedding_classifier/data/filtered/composition/foreground/n04310018_1941.JPEG"
#         elif foreground == 'rectangleTriangle':
#             source_path = "/<dirname>/<name>/g-t_embedding_classifier/data/filtered/composition/foreground/rectangleTriangle.png"

#         # --- Background paths ---
#         BASE_BG_PATH = "/<dirname>/<name>/g-t_embedding_classifier/data/filtered/composition/background"
#         if background == 'fish':
#             target_path = os.path.join(BASE_BG_PATH, "n01443537_223.JPEG")
#         elif background == 'indoor':
#             target_path = os.path.join(BASE_BG_PATH, "n03761084_821.JPEG")
#         elif background == 'building':
#             target_path = os.path.join(BASE_BG_PATH, "n03781244_3005.JPEG")
#         elif background == 'beach':
#             target_path = os.path.join(BASE_BG_PATH, "n03888257_482.JPEG")
#         elif background in synthetic_backgrounds:
#             target_path = os.path.join(synthetic_bg_dir, f"{background}.png")
            
#         radius = 800 if foreground == 'rectangleTriangle' else 170

#         output_filename = f"{foreground}_on_{background}.png"
#         output_dir = "/<dirname>/<name>/g-t_embedding_classifier/data/filtered/composition/composed_rotated/1_degree_FIXED" 
#         os.makedirs(output_dir, exist_ok=True)
        

#         extract_and_superimpose_circle_all_rotations_counterrot(
#             source_path=source_path,
#             shape=foreground,
#             target_path=target_path,
#             radius=radius,
#             output_dir=output_dir,rot_option=rot_option,
#             output_filename=output_filename
#         )
import os
import cv2
import numpy as np
from PIL import Image

# ------------------------------------------------------------
# UTILITY FUNCTIONS
# ------------------------------------------------------------
def pad_to_square(image):
    """Pads an image to a square shape using black borders."""
    h, w = image.shape[:2]
    max_dim = max(h, w)
    
    # Calculate padding amounts
    pad_h = max_dim - h
    pad_w = max_dim - w
    
    # Pad top/bottom, left/right equally
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    # Pad the image
    padded_image = cv2.copyMakeBorder(
        image, top, bottom, left, right, 
        cv2.BORDER_CONSTANT, value=[0, 0, 0] # Black border
    )
    
    return padded_image

def generate_synthetic_background(type_name, size=(375, 500), line_thickness=3, cell_size=50):
    """Generate a synthetic background based on the given type."""
    h, w = size
    bg = np.ones((h, w, 3), dtype=np.uint8) * 255  # white base

    if type_name == "chessboard":
        for y in range(0, h, cell_size):
            for x in range(0, w, cell_size):
                if ((x // cell_size) + (y // cell_size)) % 2 == 0:
                    bg[y:y+cell_size, x:x+cell_size] = 0  # black

    elif type_name == "grid_lines":
        for y in range(0, h, cell_size):
            cv2.line(bg, (0, y), (w, y), (0, 0, 0), line_thickness)
        for x in range(0, w, cell_size):
            cv2.line(bg, (x, 0), (x, h), (0, 0, 0), line_thickness)

    elif type_name == "horizontal_lines":
        for y in range(0, h, cell_size):
            cv2.line(bg, (0, y), (w, y), (0, 0, 0), line_thickness)

    elif type_name == "vertical_lines":
        for x in range(0, w, cell_size):
            cv2.line(bg, (x, 0), (x, h), (0, 0, 0), line_thickness)

    else:
        # Fallback to a solid color if type is unknown
        bg[:] = (200, 200, 200)

    return bg

# ------------------------------------------------------------
# MAIN COMPOSITING FUNCTION (Modified)
# ------------------------------------------------------------
def extract_and_superimpose_circle_all_rotations_counterrot(
    source_path, shape, target_path, radius, output_dir, rot_option, output_filename
):
    print("--- Starting Counter-Rotation Process ---")
    output_base_name = output_filename.split('/')[-1].split('.')[0]
    output_subdir_name = output_base_name + "_" + rot_option

    # Load images
    src_orig = cv2.imread(source_path, cv2.IMREAD_COLOR)
    tgt_orig = cv2.imread(target_path, cv2.IMREAD_COLOR)
    
    # Pad source image to be square for perfect circle mask
    src = pad_to_square(src_orig)
    print(f"Padded Source image shape (H, W, C): {src.shape}")

    src_center = (src.shape[1] // 2, src.shape[0] // 2)
    tgt_center = (tgt_orig.shape[1] // 2, tgt_orig.shape[0] // 2)

    # Gradual alpha mask for smooth blending
    mask = np.zeros(src.shape[:2], dtype=np.float32)
    cv2.circle(mask, src_center, radius, 1.0, -1) 

    inner_radius_factor = 0.8
    feather_width = radius * (1 - inner_radius_factor)
    inner_radius = int(radius * inner_radius_factor)

    ORIGINAL_OPAQUE_DIAMETER = 2 * inner_radius
    print(f"Original (Unscaled) Opaque Foreground Diameter: {ORIGINAL_OPAQUE_DIAMETER} pixels")

    for y in range(src.shape[0]):
        for x in range(src.shape[1]):
            dist = np.sqrt((x - src_center[0])**2 + (y - src_center[1])**2)
            if dist > inner_radius:
                alpha_val = 1.0 - (dist - inner_radius) / feather_width
                mask[y, x] = max(0.0, alpha_val)

    src_float = src.astype(np.float32) / 255.0
    mask_3ch = cv2.merge([mask] * 3)
    src_circle = (src_float * mask_3ch * 255).astype(np.uint8)

    # --- FIX: FORCING PATCH SIZE TO 2 * RADIUS (340x340 for radius=170) ---
    
    # We define the crop dimension by the radius, guaranteeing a 2*radius patch
    crop_dim = radius 

    x, y = src_center
    # Crop boundaries ensure a patch of size (x + crop_dim) - (x - crop_dim) = 2*crop_dim
    x1, y1 = max(0, x - crop_dim), max(0, y - crop_dim)
    x2, y2 = min(src.shape[1], x + crop_dim), min(src.shape[0], y + crop_dim)
    
    cropped_circle = src_circle[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]
    orig_patch_size = cropped_circle.shape[0] # The dimension of the cropped square patch
    
    print(f"Cropped Circle Shape (H, W, C): {cropped_circle.shape}")
    # ---------------------------------------------------------------------
    
    tgt_h, tgt_w = tgt_orig.shape[:2]
    print(f"Background image size (H, W): {tgt_h}, {tgt_w}")

    for scale_idx in range(60):
        if scale_idx not in (0, 3, 15):
            continue

        current_scale_factor = 1.0 / (scale_idx + 1)
        
        # Calculate new patch size based on the consistent original patch size (340)
        new_dim = int(orig_patch_size * current_scale_factor)
        if new_dim % 2 != 0:
            new_dim += 1
        size = (new_dim, new_dim)

        if new_dim <= 0 or new_dim > min(tgt_w, tgt_h):
            print(f"Skipping scale {scale_idx + 1}: Resized patch size {new_dim}x{new_dim} is invalid or too large.")
            continue
            
        scaled_opaque_diameter = int(ORIGINAL_OPAQUE_DIAMETER * current_scale_factor)
        print(f"\n--- Scale Factor 1/{scale_idx + 1} ({current_scale_factor:.4f}) ---")
        print(f"Rescaled Opaque Foreground Diameter: {scaled_opaque_diameter} pixels")
        print(f"Resized Patch Dimensions (including padding): {new_dim}x{new_dim}")

        interpolation = cv2.INTER_AREA if current_scale_factor < 1.0 else cv2.INTER_LINEAR
        circle_resized = cv2.resize(cropped_circle, size, interpolation=interpolation)
        mask_resized = cv2.resize(cropped_mask, size, interpolation=cv2.INTER_LINEAR)

        subfolder = os.path.join(output_dir, output_subdir_name, f"scale{scale_idx + 1}")
        os.makedirs(subfolder, exist_ok=True)

        for init_angle in range(0, -360, -1):

            if(rot_option == "b.5Rot_f.5Rot_1degreeTotal"):
                angle = init_angle/2
            else:
                angle = init_angle
            
            # Convert resized patch and mask to PIL images for potential rotation
            pil_circle = Image.fromarray(cv2.cvtColor(circle_resized, cv2.COLOR_BGR2RGB))
            pil_mask = Image.fromarray((mask_resized * 255).astype(np.uint8))

            # --- FOREGROUND ROTATION CONTROL BLOCK ---
            if rot_option == "b.5Rot_f.5Rot_1degreeTotal":
                # Rotate foreground by 'angle'
                rotated_circle = pil_circle.rotate(angle, resample=Image.BICUBIC, expand=False)
                rotated_mask = pil_mask.rotate(angle, resample=Image.BICUBIC, expand=False)

                circle_np = cv2.cvtColor(np.array(rotated_circle), cv2.COLOR_RGB2BGR)
                mask_np = np.array(rotated_mask).astype(np.float32) / 255.0
            
            elif rot_option == "bRot_fStat":
                # Keep foreground static
                circle_np = cv2.cvtColor(np.array(pil_circle), cv2.COLOR_RGB2BGR)
                mask_np = np.array(pil_mask).astype(np.float32) / 255.0
            
            # else:
            #      # Default to static foreground if option is unknown
            #     circle_np = cv2.cvtColor(np.array(pil_circle), cv2.COLOR_RGB2BGR)
            #     mask_np = np.array(pil_mask).astype(np.float32) / 255.0
            # -------------------------------------------------------------

            mask_3ch_rotated = cv2.merge([mask_np] * 3)

            # --- Background counter-rotation (opposite direction) ---
            opposite_angle = -angle
            pil_bg = Image.fromarray(cv2.cvtColor(tgt_orig, cv2.COLOR_BGR2RGB))

            # Rotate background with expand=True using LANCZOS for higher quality resampling
            rotated_bg = pil_bg.rotate(opposite_angle, resample=3, expand=True) 
            rot_w, rot_h = rotated_bg.size

            # Crop center to original target size
            cx, cy = rot_w // 2, rot_h // 2
            left = cx - tgt_w // 2
            upper = cy - tgt_h // 2
            right = left + tgt_w
            lower = upper + tgt_h
            rotated_bg_cropped = rotated_bg.crop((left, upper, right, lower))

            # Convert to BGR for OpenCV and smooth edges
            tgt = cv2.cvtColor(np.array(rotated_bg_cropped), cv2.COLOR_RGB2BGR)
            # Apply stronger Gaussian blur to smooth serrated edges
            tgt = cv2.GaussianBlur(tgt, (3, 3), sigmaX=1.0, sigmaY=1.0) 

            # Paste foreground at exact target center
            tx, ty = tgt_center
            rot_h, rot_w = circle_np.shape[:2]
            tx1, ty1 = tx - rot_w // 2, ty - rot_h // 2
            tx2, ty2 = tx1 + rot_w, ty1 + rot_h

            roi = tgt[ty1:ty2, tx1:tx2].astype(np.float32) / 255.0
            blended = circle_np.astype(np.float32) / 255.0 * mask_3ch_rotated + roi * (1 - mask_3ch_rotated)
            tgt[ty1:ty2, tx1:tx2] = (blended * 255).astype(np.uint8)

            out_path = os.path.join(subfolder, f"{output_base_name}_{abs(init_angle)}.png")
            cv2.imwrite(out_path, tgt)

        print(f"Finished scale {scale_idx} for {output_base_name}: {abs(init_angle)} total rotations completed.")
    print("********************************************")


# ------------------------------------------------------------
# DRIVER CODE 
# ------------------------------------------------------------
foreground_list = ['dog', 'lizard', 'train', 'rectangleTriangle']
# background_list = ["beach"]
#COMMENT OUT EITHER NATURAL OR SYNTHETIC BACKGROUNDS
# background_list = ['fish', 'indoor', 'building', 'beach']
background_list = ["chessboard", "grid_lines", "horizontal_lines", "vertical_lines"]
synthetic_backgrounds = ["chessboard", "grid_lines", "horizontal_lines", "vertical_lines"]
check_combinations = False
check_foreground_only = True
#FOR SYNTHETIC BACKGROUNDS, USE SECOND LIST
#OTHERWISE, USE FIRST LIST
# required_combinations = ["lizard_on_fish","train_on_indoor"] #dog_on_beach, lizard_on_fish, train_on_indoor, dog, lizard, train
required_combinations = ["lizard","train"]

rot_options = ["bRot_fStat","b.5Rot_f.5Rot_1degreeTotal"] #b.5Rot_f.5Rot_1degreeTotal, bRot_fStat
# Directory for synthetic backgrounds
synthetic_bg_dir = "/<dirname>/<name>/g-t_embedding_classifier/data/filtered/composition/synthetic_backgrounds"
os.makedirs(synthetic_bg_dir, exist_ok=True)

# Generate and save synthetic backgrounds once
# for bg_type in synthetic_backgrounds:
#     bg_path = os.path.join(synthetic_bg_dir, f"{bg_type}.png")
#     if not os.path.exists(bg_path):
#         print(f"Generating synthetic background: {bg_type}")
#         # Using a standard small size for synthetic bgs as they are patterns
#         bg = generate_synthetic_background(bg_type, size=(375, 500)) 
#         cv2.imwrite(bg_path, bg)

for foreground in foreground_list:
    for background in background_list:
        combination = f"{foreground}_on_{background}"
        
        # Process only 'dog' combinations as suggested by the code's pattern
        if(check_combinations == True):
            if combination not in required_combinations:
                # print(f"Skipping combination: {combination}")
                # print("Required combinations are:", required_combinations)
                continue
        if(check_foreground_only == True):
            if not any(rc.split("_")[0] == foreground for rc in required_combinations):
                continue
        # if "dog" not in combination:
        #     continue
        for rot_option in rot_options:
            print(f"\n========================================================")
            print(f"Processing: {foreground} on {background} (Rotation Option: {rot_option})")
            print(f"========================================================")

            # --- Foreground paths ---
            if foreground == 'dog':
                source_path = "/<dirname>/<name>/g-t_embedding_classifier/data/filtered/composition/foreground/n02108422_4102.JPEG"
            elif foreground == 'lizard':
                source_path = "/<dirname>/<name>/g-t_embedding_classifier/data/filtered/composition/foreground/n01629819_74.JPEG"
            elif foreground == 'train':
                source_path = "/<dirname>/<name>/g-t_embedding_classifier/data/filtered/composition/foreground/n04310018_1941.JPEG"
            elif foreground == 'rectangleTriangle':
                source_path = "/<dirname>/<name>/g-t_embedding_classifier/data/filtered/composition/foreground/rectangleTriangle.png"

            # --- Background paths ---
            BASE_BG_PATH = "/<dirname>/<name>/g-t_embedding_classifier/data/filtered/composition/background"
            if background == 'fish':
                target_path = os.path.join(BASE_BG_PATH, "n01443537_223.JPEG")
            elif background == 'indoor':
                target_path = os.path.join(BASE_BG_PATH, "n03761084_821.JPEG")
            elif background == 'building':
                target_path = os.path.join(BASE_BG_PATH, "n03781244_3005.JPEG")
            elif background == 'beach':
                target_path = os.path.join(BASE_BG_PATH, "n03888257_482.JPEG")
            elif background in synthetic_backgrounds:
                target_path = os.path.join(synthetic_bg_dir, f"{background}.png")
                
            # radius = 800 if foreground == 'rectangleTriangle' else 170
            if(foreground =="dog"):
                radius = 170
            elif(foreground in ["lizard","train"]):
                radius = 166

            output_filename = f"{foreground}_on_{background}.png"
            output_dir = "/<dirname>/<name>/g-t_embedding_classifier/data/filtered/composition/composed_rotated/1_degree_FIXED" 
            os.makedirs(output_dir, exist_ok=True)
            

            extract_and_superimpose_circle_all_rotations_counterrot(
                source_path=source_path,
                shape=foreground,
                target_path=target_path,
                radius=radius,
                output_dir=output_dir,rot_option=rot_option,
                output_filename=output_filename
            )