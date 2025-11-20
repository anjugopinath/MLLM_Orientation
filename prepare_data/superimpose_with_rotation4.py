import os
import sys
import cv2
import numpy as np
from PIL import Image

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
    
    # Store the original offsets for later center calculation
    original_offsets = (left, top)
    return padded_image, original_offsets

def extract_and_superimpose_circle_all_rotations(source_path, shape, target_path, radius, output_dir, output_filename):
    output_base_name = output_filename.split('/')[-1].split('.')[0]

    # Load images
    src_orig = cv2.imread(source_path, cv2.IMREAD_COLOR)
    tgt_orig = cv2.imread(target_path, cv2.IMREAD_COLOR)

    # --- FIX 1: Pad source image to be square ---
    # This is crucial for ensuring the extracted circle is not an oval if the source image is non-square.
    src, original_offsets = pad_to_square(src_orig)
    
    print(f"Original Source image shape (H, W, C): {src_orig.shape}")
    print(f"Padded Source image shape (H, W, C): {src.shape}")
    print(f"Target image shape (H, W, C): {tgt_orig.shape}")

    # Get centers based on the square padded image
    src_center = (src.shape[1] // 2, src.shape[0] // 2)
    tgt_center = (tgt_orig.shape[1] // 2, tgt_orig.shape[0] // 2)

    print(f"Padded Source center (x, y): {src_center}")
    print(f"Target center (x, y): {tgt_center}")

    # --- Create a gradual alpha mask ---
    mask = np.zeros(src.shape[:2], dtype=np.float32) # Use float32 for mask

    # Define feathering parameters
    inner_radius_factor = 0.8 
    feather_width = radius * (1 - inner_radius_factor)
    inner_radius = int(radius * inner_radius_factor)
    # opaque_diameter = 2 * inner_radius
    # print(f"Original Radius (Total Circle): {radius}")
    # print(f"Inner Radius Factor: {inner_radius_factor}")
    # print(f"Opaque Circle Diameter: {opaque_diameter}")
    # Calculate the original (unscaled) opaque diameter once
    ORIGINAL_OPAQUE_DIAMETER = 2 * inner_radius 
    print(f"Original Radius (Total Circle): {radius}")
    print(f"Inner Radius Factor: {inner_radius_factor}")
    print(f"Unscaled Opaque Circle Diameter: {ORIGINAL_OPAQUE_DIAMETER}")

    # Apply a circular gradient
    for y in range(src.shape[0]):
        for x in range(src.shape[1]):
            dist = np.sqrt((x - src_center[0])**2 + (y - src_center[1])**2)
            if dist <= inner_radius:
                mask[y, x] = 1.0 # Fully opaque
            elif dist < radius:
                # Calculate alpha based on distance from inner_radius to radius
                alpha_val = 1.0 - (dist - inner_radius) / feather_width
                mask[y, x] = max(0.0, alpha_val) 
            else:
                mask[y, x] = 0.0 # Fully transparent

    print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")

    # Save mask for visualization
    cv2.imwrite("gradual_circular_mask.png", (mask * 255).astype(np.uint8))

    # Extract circular region 
    src_float = src.astype(np.float32) / 255.0
    mask_3ch = cv2.merge([mask] * 3) 
    src_circle_masked = src_float * mask_3ch # This is the full padded, masked image

    # Crop bounding box: This must be a square bounding box centered at src_center
    x, y = src_center
    
    # The cropped region must be 2*radius x 2*radius for the rotation/resize logic to work cleanly
    # We choose the bounding box that tightly encloses the circle
    x1, y1 = x - radius, y - radius
    x2, y2 = x + radius, y + radius
    
    # Ensure crop coordinates are within the padded image bounds (though they should be if radius is sensible)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(src.shape[1], x2)
    y2 = min(src.shape[0], y2)

    # If the radius is large enough, the cropped region will be 2*radius x 2*radius
    cropped_circle_float = src_circle_masked[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2] 
    
    # Convert back to uint8 for saving/rotation
    cropped_circle = (cropped_circle_float * 255).astype(np.uint8)
    
    cv2.imwrite("cropped_circle_feathered.png", cropped_circle)
    cv2.imwrite("cropped_mask_feathered.png", (cropped_mask * 255).astype(np.uint8))
    
    print(f"Cropped Circle Shape (H, W, C): {cropped_circle.shape}")

    tgt_h, tgt_w = tgt_orig.shape[:2]
    print(f"Target image size (H, W): {tgt_h}, {tgt_w}")

    # The size of the unresized cropped image patch should be 2*radius x 2*radius
    orig_patch_size = cropped_circle.shape[0]

    for scale_idx in range(60):
        if scale_idx not in (0, 3, 15):
            continue

        '''
        Resizes a circular image patch (and its mask).
        Rotates it from 0° to -359°.
        Blends the rotated patch into the target image at the center.
        Saves the resulting image for each angle and scale.
        '''

        current_scale_factor = 1.0 / (scale_idx + 1)

        # --- NEW CALCULATION AND PRINT STATEMENT HERE ---
        scaled_opaque_diameter = int(ORIGINAL_OPAQUE_DIAMETER * current_scale_factor)
        # Ensure it's even, matching the logic for new_dim
        if scaled_opaque_diameter % 2 != 0:
            scaled_opaque_diameter += 1
        
        print(f"Scale {scale_idx + 1} Factor: {current_scale_factor:.4f}")
        print(f"Scaled Opaque Diameter: {scaled_opaque_diameter}")
        # -----------------------------------------------------------------
        
        # Calculate new size based on original patch size * scale factor
        new_dim = int(orig_patch_size * current_scale_factor)
        
        # Ensure the new dimensions are even for clean centering/sizing
        if new_dim % 2 != 0:
            new_dim += 1
            
        # The new patch size for resize
        size = (new_dim, new_dim)
        new_half_dim = new_dim // 2

        # Check for target bounds
        if new_half_dim * 2 > min(tgt_w, tgt_h):
             print(f"Skipping scale {scale_idx + 1}: Resized patch size {new_dim}x{new_dim} is too large for target {tgt_w}x{tgt_h}.")
             continue

        print("scale_idx:", scale_idx)
        print("new_dim:", new_dim)

        # Resize cropped circle & mask
        if size[0] > 0 and size[1] > 0:
            # Use INTER_AREA for shrinking, INTER_LINEAR/CUBIC for growing
            interpolation = cv2.INTER_AREA if current_scale_factor < 1.0 else cv2.INTER_LINEAR
            circle_resized = cv2.resize(cropped_circle, size, interpolation=interpolation)
            # Use INTER_LINEAR for mask to get smoother alpha values
            mask_resized = cv2.resize(cropped_mask, size, interpolation=cv2.INTER_LINEAR)
            
            print("Resized circle shape:", circle_resized.shape)
            print("Resized mask shape:", mask_resized.shape)
        else:
            print(f"Skipping scale {scale_idx + 1}: Invalid resize size {size}.")
            continue


        # Create output subfolder for scale
        subfolder = os.path.join(output_dir, output_base_name, f"scale{scale_idx + 1}")
        os.makedirs(subfolder, exist_ok=True)

        for angle in range(0, -360, -1): 
            tgt = tgt_orig.copy()

            # Rotate with PIL (expand=False keeps the output size the same as input size)
            pil_circle = Image.fromarray(cv2.cvtColor(circle_resized, cv2.COLOR_BGR2RGB)) 
            pil_mask = Image.fromarray((mask_resized * 255).astype(np.uint8)) 

            # NOTE: expand=False means the output size is the same as the input size (new_dim x new_dim)
            rotated_circle = pil_circle.rotate(angle, resample=Image.BICUBIC, expand=False) 
            rotated_mask = pil_mask.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=0)

            # Back to NumPy
            circle_np = cv2.cvtColor(np.array(rotated_circle), cv2.COLOR_RGB2BGR) 
            mask_np = np.array(rotated_mask).astype(np.float32) / 255.0
            mask_3ch_rotated = cv2.merge([mask_np] * 3)

            # Calculate paste location based on new_half_dim
            rot_h, rot_w = circle_np.shape[:2]
            
            # Target center
            tx, ty = tgt_center 
            
            # Top-left corner of the paste area on the target image
            tx1 = tx - rot_w // 2
            ty1 = ty - rot_h // 2
            tx2 = tx1 + rot_w
            ty2 = ty1 + rot_h

            # Check bounds (already checked for the worst case, but good to keep)
            if tx1 < 0 or ty1 < 0 or tx2 > tgt_w or ty2 > tgt_h:
                 print(f"Skipping scale {scale_idx + 1}, angle {angle}: out of bounds after rotation. Paste area: ({tx1},{ty1}) to ({tx2},{ty2}). Target size: ({tgt_w},{tgt_h})")
                 continue

            # Region of Interest (ROI) from the target image
            roi = tgt[ty1:ty2, tx1:tx2].astype(np.float32) / 255.0
            
            # Double-check dimensions for blend (should match since expand=False)
            if roi.shape != circle_np.shape or roi.shape[:2] != mask_3ch_rotated.shape[:2]:
                 print(f"FATAL: Dimension mismatch for blend at scale {scale_idx + 1}, angle {angle}. ROI shape: {roi.shape}, Rotated Circle shape: {circle_np.shape}, Rotated Mask shape: {mask_3ch_rotated.shape}")
                 continue

            # Alpha Blending Formula: (Foreground * Alpha) + (Background * (1 - Alpha))
            blended = circle_np.astype(np.float32) / 255.0 * mask_3ch_rotated + roi * (1.0 - mask_3ch_rotated)
            
            # Paste the blended region back into the target image
            tgt[ty1:ty2, tx1:tx2] = (blended * 255).astype(np.uint8)
            
            if(angle==0):
                 print(f"ROI shape: {roi.shape}, Rotated Circle shape: {circle_np.shape}, Rotated Mask shape: {mask_3ch_rotated.shape}")
                 print("blended shape:", blended.shape)
                 print("tgt shape after blending:", tgt.shape)

            final = (tgt).astype(np.uint8) 
            out_path = os.path.join(subfolder, f"{output_base_name}_{abs(angle)}.png")
            
            cv2.imwrite(out_path, final)

            # print(f"Saved: {out_path}")
        print("********************************************")

# ---------- DRIVER CODE ----------

foreground_list = ['dog', 'lizard', 'train', 'rectangleTriangle']
background_list = ['fish', 'indoor', 'building', 'beach']

# Replace with your actual paths
BASE_PATH = "/<dirname>/<name>/g-t_embedding_classifier/data/filtered/composition/"
OUTPUT_PATH = "/<dirname>/<name>/g-t_embedding_classifier/data/filtered/composition/composed_rotated/1_degree_FIXED"
os.makedirs(OUTPUT_PATH, exist_ok=True)


for foreground in foreground_list:
    for background in background_list:
        combination = f"{foreground}_on_{background}"
        if(combination != 'train_on_indoor'):
            continue
        
        print(f"Processing: {foreground} on {background}")
        
        # Set foreground path
        if foreground == 'dog':
            source_path = os.path.join(BASE_PATH, "foreground/n02108422_4102.JPEG")
            shape = "dog"
        elif foreground == 'lizard':
            source_path = os.path.join(BASE_PATH, "foreground/n01629819_74.JPEG")
            shape = "lizard"
        elif foreground == 'train':
            source_path = os.path.join(BASE_PATH, "foreground/n04310018_1941.JPEG")
            shape = "train"
        elif foreground == 'rectangleTriangle':
            source_path = os.path.join(BASE_PATH, "foreground/rectangleTriangle.png")
            shape = "rectangleTriangle"

        # Set background path
        if background == 'fish':
            target_path = os.path.join(BASE_PATH, "background/n01443537_223.JPEG")
        elif background == 'indoor':
            target_path = os.path.join(BASE_PATH, "background/n03761084_821.JPEG")
        elif background == 'building':
            target_path = os.path.join(BASE_PATH, "background/n03781244_3005.JPEG")
        elif background == 'beach':
            target_path = os.path.join(BASE_PATH, "background/n03888257_482.JPEG")

        if foreground == 'rectangleTriangle':
            # Use a smaller radius if the image is very large, for testing
            radius = 170 
        else:
            if(combination in ['lizard_on_fish', 'train_on_indoor']):
                radius = 166
            else:
                radius = 170

        output_filename = f"{foreground}_on_{background}.png"
        
        extract_and_superimpose_circle_all_rotations(
            source_path=source_path,
            shape=shape,
            target_path=target_path,
            radius=radius,
            output_dir=OUTPUT_PATH,
            output_filename=output_filename
        )