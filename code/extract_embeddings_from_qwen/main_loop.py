# import torch
# import json
# import os
# import numpy as np
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
# from qwen_vl_utils import process_vision_info

# # LD_PRELOAD="$CONDA_PREFIX/lib/libittnotify.so" python main_loop.py

# category = "navi" # "navi" or "coco"
# img_id = "002"
# # Setup paths
# if(category=="coco"):
#     vision_output_path = "/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/llava/LLaVA/llava/eval/pca_images/what_angle/1_degree_FIXED/coco-qwen"
#     text_output_path = "/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/generate_files_for_llava/qs_list/coco-qwen/output_ans"
# elif(category=="navi"):

#     # final_vision_output_path = "/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/llava/LLaVA/llava/eval/pca_images/what_angle/1_degree_FIXED/navi-qwen/firetruck_000_002"
#     if(img_id=="000"):
#         vision_output_path = "/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/llava/LLaVA/llava/eval/pca_images/what_angle/1_degree_FIXED/navi-qwen/firetruck_000"
#         text_output_path = "/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/generate_files_for_llava/qs_list/navi-qwen/output_ans/firetruck_000"
#     elif(img_id=="002"):
#         vision_output_path = "/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/llava/LLaVA/llava/eval/pca_images/what_angle/1_degree_FIXED/navi-qwen/firetruck_002"
#         text_output_path = "/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/generate_files_for_llava/qs_list/navi-qwen/output_ans/firetruck_002"

# os.makedirs(vision_output_path, exist_ok=True)
# os.makedirs(text_output_path, exist_ok=True)


# # 1. Load Model
# model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     model_id, torch_dtype="auto", device_map="auto"
# )
# processor = AutoProcessor.from_pretrained(model_id)

# # Storage for results
# text_results = {}
# vision_embeddings_dict = {}

# if(category == "coco"):
#     # Reference image (0 degrees)
#     ref_img = "/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/utils/natural_images/coco-dataset/output/000000500634/scale1/000000500634_0.png"
# elif(category == "navi"):
#     if(img_id=="000"):
#         ref_img = "/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/utils/natural_images/navi-dataset/output/fireengine/scale1/fire_engine_toy_red_yellow_s_000_0.png"
#     elif(img_id=="002"):
#         ref_img = "/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/utils/natural_images/navi-dataset/output/fireengine/scale1/fire_engine_toy_red_yellow_s_002_0.png"

# # 2. Loop through 1 to 359 degrees
# for degree in range(1, 360):
#     if(category == "coco"):
#         test_img = f"/s/red/a/nobackup/visiogn/anju/g-t_embedding_classifier/utils/natural_images/coco-dataset/output/000000500634/scale1/000000500634_{degree}.png"
#     elif(category == "navi"):
#         if(img_id=="000"):
#             test_img = f"/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/utils/natural_images/navi-dataset/output/fireengine/scale1/fire_engine_toy_red_yellow_s_000_{degree}.png"
#         elif(img_id=="002"):
#             test_img = f"/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/utils/natural_images/navi-dataset/output/fireengine/scale1/fire_engine_toy_red_yellow_s_002_{degree}.png"
#     if not os.path.exists(test_img):
#         continue

#     if(category=="coco"):
#         messages = [{
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": ref_img},
#                 {"type": "image", "image": test_img},
#                 {"type": "text", "text": f"In Image 1, there is a bicycle leaning against a wall. Image 2 is the same scene, but the bicycle has been rotated clockwise. Based on Image 1, precisely how many degrees clockwise has the bicycle been rotated in Image 2? Provide an estimate between 1 and 359."}
#             ]
#         }]
#     elif(category=="navi"):
#         messages = [{
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": ref_img},
#                 {"type": "image", "image": test_img},
#                 {"type": "text", "text": f"In Image 1, there is a red and yellow toy fireengine placed on a table(0 degrees). Image 2 is the same scene, but the toy fireengine has been rotated clockwise. Based on Image 1, precisely how many degrees clockwise has the toy fireengine been rotated in Image 2? Provide an estimate between 1 and 359."}
#             ]
#         }]

#     # Preparation
#     text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     image_inputs, video_inputs = process_vision_info(messages)
#     inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")

#     # Generate Text Output
#     output = model.generate(**inputs, max_new_tokens=50, return_dict_in_generate=True)
#     generated_ids = output.sequences
#     mllm_answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
#     # Store text result
#     text_results[str(degree)] = mllm_answer

#     # Extract Vision Embeddings
#     with torch.no_grad():
#         vision_outputs = model.model.visual(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])
#         # We store the last hidden state of the test image (Image 2)
#         # Note: Qwen2.5-VL concatenates tokens. We extract the second half.
#         full_embeds = vision_outputs.last_hidden_state
#         num_patches = full_embeds.shape[0] // 2 
#         test_image_embeds = full_embeds[num_patches:].to(torch.float32).cpu().numpy()
        
#         vision_embeddings_dict[str(degree)] = test_image_embeds

#     print(f"Processed {degree}°: {mllm_answer[:30]}...")

# # 3. Save Files
# with open(os.path.join(text_output_path, "rotation_responses.json"), "w") as f:
#     json.dump(text_results, f, indent=4)

# np.save(os.path.join(vision_output_path, "vision_embeddings.npy"), vision_embeddings_dict)

# print("✅ Data collection complete.")
import torch
import json
import os
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# --- CONFIGURATION ---
category = "navi"  # Options: "navi" or "coco"
img_id = "000"     # Set to "000" first, then "002" to enable concatenation

# --- PATH SETUP ---
if category == "coco":
    vision_output_path = "/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/llava/LLaVA/llava/eval/pca_images/what_angle/1_degree_FIXED/coco-qwen"
    text_output_path = "/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/generate_files_for_llava/qs_list/coco-qwen/output_ans"
elif category == "navi":
    final_vision_output_path = "/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/llava/LLaVA/llava/eval/pca_images/what_angle/1_degree_FIXED/navi-qwen/firetruck_000_002"
    if img_id == "000":
        vision_output_path = "/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/llava/LLaVA/llava/eval/pca_images/what_angle/1_degree_FIXED/navi-qwen/firetruck_000"
        text_output_path = "/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/generate_files_for_llava/qs_list/navi-qwen/output_ans/firetruck_000"
    elif img_id == "002":
        vision_output_path = "/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/llava/LLaVA/llava/eval/pca_images/what_angle/1_degree_FIXED/navi-qwen/firetruck_002"
        text_output_path = "/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/generate_files_for_llava/qs_list/navi-qwen/output_ans/firetruck_002"

os.makedirs(vision_output_path, exist_ok=True)
os.makedirs(text_output_path, exist_ok=True)

# --- 1. MODEL INITIALIZATION ---
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)

# Storage
text_results = {}
vision_embeddings_dict = {}
multi_view_dict = {}

# Define Object Names and Reference Paths
if category == "coco":
    ref_img = "/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/utils/natural_images/coco-dataset/output/000000500634/scale1/000000500634_0.png"
    prompt_obj = "bicycle"
elif category == "navi":
    prompt_obj = "red and yellow toy fireengine"
    if img_id == "000":
        ref_img = "/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/utils/natural_images/navi-dataset/output/fireengine/scale1/fire_engine_toy_red_yellow_s_000_0.png"
    else:
        ref_img = "/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/utils/natural_images/navi-dataset/output/fireengine/scale1/fire_engine_toy_red_yellow_s_002_0.png"

# --- 2. STEP 0: EXTRACT & STORE 0° REFERENCE EMBEDDING ---
print(f"--- Extracting reference (0°) embedding for {img_id} ---")
msg_ref = [{"role": "user", "content": [{"type": "image", "image": ref_img}]}]
img_in_ref, _ = process_vision_info(msg_ref)
vis_in_ref = processor(text=[""], images=img_in_ref, return_tensors="pt").to("cuda")

with torch.no_grad():
    v_out_ref = model.model.visual(vis_in_ref["pixel_values"], grid_thw=vis_in_ref["image_grid_thw"])
    ref_features = v_out_ref.last_hidden_state.mean(axis=0).to(torch.float32).cpu().numpy()
    vision_embeddings_dict["0"] = ref_features

# Multi-view check for 0 degrees
if category == "navi" and img_id == "002":
    path_000 = vision_output_path.replace("firetruck_002", "firetruck_000")
    file_000 = os.path.join(pagth_000, "vision_embeddings.npy")
    if os.path.exists(file_000):
        data_000 = np.load(file_000, allow_pickle=True).item()
        if "0" in data_000:
            multi_view_dict["0"] = np.concatenate([data_000["0"], ref_features])

# --- 3. THE LOOP (1 to 359) ---
for degree in range(1, 360):
    if category == "coco":
        test_img = f"/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/utils/natural_images/coco-dataset/output/000000500634/scale1/000000500634_{degree}.png"
    else:
        test_img = f"/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/utils/natural_images/navi-dataset/output/fireengine/scale1/fire_engine_toy_red_yellow_s_{img_id}_{degree}.png"

    if not os.path.exists(test_img):
        continue

    # --- PART A: TEXT GENERATION (Compare 0° vs Degree) ---
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": ref_img},
            {"type": "image", "image": test_img},
            {"type": "text", "text": f"Image 1 shows the {prompt_obj} at 0 degrees. Image 2 is the same scene rotated clockwise. Exactly how many degrees clockwise is Image 2? (1-359)"}
        ]
    }]
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[chat_text], images=image_inputs, padding=True, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)
        mllm_answer = processor.batch_decode(output, skip_special_tokens=True)[0]
    text_results[str(degree)] = mllm_answer

    # --- PART B: CLEAN FEATURE EXTRACTION (Single Image) ---
    msg_single = [{"role": "user", "content": [{"type": "image", "image": test_img}]}]
    img_in_single, _ = process_vision_info(msg_single)
    vis_in_single = processor(text=[""], images=img_in_single, return_tensors="pt").to("cuda")

    with torch.no_grad():
        v_out_single = model.model.visual(vis_in_single["pixel_values"], grid_thw=vis_in_single["image_grid_thw"])
        clean_features = v_out_single.last_hidden_state.mean(axis=0).to(torch.float32).cpu().numpy()
        vision_embeddings_dict[str(degree)] = clean_features

    # --- PART C: MULTI-VIEW CONCATENATION ---
    if category == "navi" and img_id == "002":
        if os.path.exists(file_000):
            # data_000 already loaded in Step 0
            if str(degree) in data_000:
                multi_view_dict[str(degree)] = np.concatenate([data_000[str(degree)], clean_features])

    if degree % 10 == 0:
        print(f"Progress: {degree}°/359° complete.")

# --- 4. DATA PERSISTENCE ---
# Save Text Answers
with open(os.path.join(text_output_path, "rotation_responses.json"), "w") as f:
    json.dump(text_results, f, indent=4)

# Save Single-View Embeddings (includes '0')
np.save(os.path.join(vision_output_path, "vision_embeddings.npy"), vision_embeddings_dict)

# Save Concatenated Embeddings
if multi_view_dict:
    os.makedirs(final_vision_output_path, exist_ok=True)
    np.save(os.path.join(final_vision_output_path, "multi_view_embeddings.npy"), multi_view_dict)

print(f"✅ Data collection complete for {img_id}. Total angles stored: {len(vision_embeddings_dict)}")