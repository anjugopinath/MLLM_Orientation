import os
import sys
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

# ------------------------------------------------------------
# SYNTHETIC BACKGROUND GENERATOR (Unchanged)
# ------------------------------------------------------------
def generate_synthetic_background(type_name, size=(1024, 1024), line_thickness=3, cell_size=100):
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
        raise ValueError(f"Unknown background type: {type_name}")

    return bg

# ------------------------------------------------------------
# MAIN COMPOSITING FUNCTION (Modified with print statements)
# ------------------------------------------------------------
def extract_and_superimpose_circle_all_rotations(source_path, shape, target_path, radius, output_dir, output_filename):
    print("--- Starting Superimposition Process ---")
    output_base_name = output_filename.split('/')[-1].split('.')[0]

    # Load images
    src_orig = cv2.imread(source_path, cv2.IMREAD_COLOR)
    tgt_orig = cv2.imread(target_path, cv2.IMREAD_COLOR)

    # --- NEW LOGIC: Pad source image to be square ---
    src = pad_to_square(src_orig)
    
    print(f"Padded Source image shape (H, W, C): {src.shape}")
    
    # --- ADDED PRINT: Background image size ---
    tgt_h, tgt_w = tgt_orig.shape[:2]
    print(f"Background image size (H, W): {tgt_h}, {tgt_w}") 
    # -----------------------------------------------

    # Center is calculated based on the square padded image
    src_center = (src.shape[1] // 2, src.shape[0] // 2)
    tgt_center = (tgt_orig.shape[1] // 2, tgt_orig.shape[0] // 2)

    # Gradual alpha mask
    mask = np.zeros(src.shape[:2], dtype=np.float32) 
    
    # Define feathering parameters
    inner_radius_factor = 0.8
    feather_width = radius * (1 - inner_radius_factor)
    inner_radius = int(radius * inner_radius_factor)
    
    # --- ADDED PRINT: Opaque foreground diameter ---
    opaque_diameter = 2 * inner_radius
    print(f"Original (Unscaled) Opaque Foreground Diameter: {opaque_diameter}")
    # -----------------------------------------------
    ORIGINAL_OPAQUE_DIAMETER = opaque_diameter # Store for scaling calculation

    # Apply circular gradient logic
    for y in range(src.shape[0]):
        for x in range(src.shape[1]):
            dist = np.sqrt((x - src_center[0])**2 + (y - src_center[1])**2)
            if dist <= inner_radius:
                mask[y, x] = 1.0 # Fully opaque
            elif dist < radius:
                alpha_val = 1.0 - (dist - inner_radius) / feather_width
                mask[y, x] = max(0.0, alpha_val) 
            else:
                mask[y, x] = 0.0 # Fully transparent

    src_float = src.astype(np.float32) / 255.0
    mask_3ch = cv2.merge([mask] * 3)
    src_circle_masked = src_float * mask_3ch

    # Crop bounding box
    x, y = src_center
    x1, y1 = x - radius, y - radius
    x2, y2 = x + radius, y + radius
    
    # Ensure crop coordinates are within bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(src.shape[1], x2)
    y2 = min(src.shape[0], y2)

    cropped_circle_float = src_circle_masked[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2] 
    cropped_circle = (cropped_circle_float * 255).astype(np.uint8)
    
    orig_patch_size = cropped_circle.shape[0]

    # The loop for scaling and rotation
    for scale_idx in range(60):
        if scale_idx not in (0, 3, 15):
            continue

        current_scale_factor = 1.0 / (scale_idx + 1)
        
        # Calculate new size based on original patch size * scale factor
        new_dim = int(orig_patch_size * current_scale_factor)
        
        # Ensure the new dimensions are even for clean centering/sizing
        if new_dim % 2 != 0:
            new_dim += 1
            
        size = (new_dim, new_dim)

        if new_dim <= 0 or new_dim > min(tgt_w, tgt_h):
            print(f"Skipping scale {scale_idx + 1}: Resized patch size {new_dim}x{new_dim} is invalid or too large.")
            continue

        # --- ADDED PRINT: Rescaled opaque diameter ---
        scaled_opaque_diameter = int(ORIGINAL_OPAQUE_DIAMETER * current_scale_factor)
        print(f"\n--- Scale Factor 1/{scale_idx + 1} ({current_scale_factor:.4f}) ---")
        print(f"Rescaled Opaque Foreground Diameter: {scaled_opaque_diameter} pixels")
        print(f"Resized Patch Dimensions: {new_dim}x{new_dim}")
        # -----------------------------------------------

        # Resize cropped circle & mask
        interpolation = cv2.INTER_AREA if current_scale_factor < 1.0 else cv2.INTER_LINEAR
        circle_resized = cv2.resize(cropped_circle, size, interpolation=interpolation)
        mask_resized = cv2.resize(cropped_mask, size, interpolation=cv2.INTER_LINEAR)
        
        # Create output subfolder for scale
        subfolder = os.path.join(output_dir, output_base_name, f"scale{scale_idx + 1}")
        os.makedirs(subfolder, exist_ok=True)

        for angle in range(0, -360, -1):
            tgt = tgt_orig.copy()
            
            # Rotate using PIL
            pil_circle = Image.fromarray(cv2.cvtColor(circle_resized, cv2.COLOR_BGR2RGB))
            pil_mask = Image.fromarray((mask_resized * 255).astype(np.uint8))
            rotated_circle = pil_circle.rotate(angle, resample=Image.BICUBIC, expand=False)
            rotated_mask = pil_mask.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=0)

            # Back to NumPy
            circle_np = cv2.cvtColor(np.array(rotated_circle), cv2.COLOR_RGB2BGR)
            mask_np = np.array(rotated_mask).astype(np.float32) / 255.0
            mask_3ch_rotated = cv2.merge([mask_np] * 3)

            rot_w, rot_h = circle_np.shape[1], circle_np.shape[0]
            tx, ty = tgt_center
            tx1, ty1 = tx - rot_w // 2, ty - rot_h // 2
            tx2, ty2 = tx1 + rot_w, ty1 + rot_h

            if tx1 < 0 or ty1 < 0 or tx2 > tgt_w or ty2 > tgt_h:
                continue

            roi = tgt[ty1:ty2, tx1:tx2].astype(np.float32) / 255.0
            if roi.shape != circle_np.shape or roi.shape[:2] != mask_3ch_rotated.shape[:2]:
                continue

            # Alpha Blending
            blended = circle_np.astype(np.float32) / 255.0 * mask_3ch_rotated + roi * (1.0 - mask_3ch_rotated)
            tgt[ty1:ty2, tx1:tx2] = (blended * 255).astype(np.uint8)

            out_path = os.path.join(subfolder, f"{output_base_name}_{abs(angle)}.png")
            cv2.imwrite(out_path, tgt)


# ------------------------------------------------------------
# DRIVER CODE (Modified to only process 'dog' and synthetic backgrounds)
# ------------------------------------------------------------
foreground_list = ['lizard', 'train'] # dog, lizard, train
synthetic_backgrounds = ["chessboard", "grid_lines", "horizontal_lines", "vertical_lines"]
BASE_PATH = "/<dirname>/<name>/g-t_embedding_classifier/data/filtered/composition/"

# Directory for synthetic backgrounds
synthetic_bg_dir = "/<dirname>/<name>/g-t_embedding_classifier/data/filtered/composition/synthetic_backgrounds"
os.makedirs(synthetic_bg_dir, exist_ok=True)

# Generate and save synthetic backgrounds once
# for bg_type in synthetic_backgrounds:
#     bg_path = os.path.join(synthetic_bg_dir, f"{bg_type}.png")
#     if not os.path.exists(bg_path):
#         print(f"Generating synthetic background: {bg_type}")
#         # Assuming the target size is 375x500 as in the original code
#         bg = generate_synthetic_background(bg_type, size=(375, 500)) 
#         cv2.imwrite(bg_path, bg)

# Now reuse cached backgrounds
for foreground in foreground_list:
    for bg_type in synthetic_backgrounds:
        combination = f"{foreground}_on_{bg_type}"
        
        print(f"\n========================================================")
        print(f"Processing: {combination}")
        print(f"========================================================")

        # Foreground setup (only dog is necessary here)
        if foreground == 'dog':
            source_path = os.path.join(BASE_PATH,"foreground/n02108422_4102.JPEG")
            shape = "dog"
        elif foreground == 'lizard':
            source_path = os.path.join(BASE_PATH, "foreground/n01629819_74.JPEG")
            shape = "lizard"
        elif foreground == 'train':
            source_path = os.path.join(BASE_PATH, "foreground/n04310018_1941.JPEG")
            shape = "train"

        target_path = os.path.join(synthetic_bg_dir, f"{bg_type}.png")
        if(foreground == 'dog'):
            radius = 170 # Use the radius defined for dog
        elif(foreground == 'lizard' or foreground == 'train'):
            radius = 166 # Use the radius defined for lizard and train:

        output_filename = f"{foreground}_on_{bg_type}.png"
        output_dir = "/<dirname>/<name>/g-t_embedding_classifier/data/filtered/composition/composed_rotated/1_degree_FIXED"
        os.makedirs(output_dir, exist_ok=True)

        extract_and_superimpose_circle_all_rotations(
            source_path=source_path,
            shape=shape,
            target_path=target_path,
            radius=radius,
            output_dir=output_dir,
            output_filename=output_filename
        )