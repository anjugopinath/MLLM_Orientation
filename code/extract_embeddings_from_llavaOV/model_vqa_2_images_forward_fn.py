import torch
import copy
from PIL import Image
import numpy as np
import os
import json

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.shared_data import embedding_outputs_global

# ─── CONFIG ───────────────────────────────────────────────
MODEL_PATH = "lmms-lab/llava-onevision-qwen2-7b-ov"
save_result = False
TEMPERATURE = 0.2
TOP_P = 1.0
dataset_category = "natural_cropped_gaussian_blur" #natural_cropped, "in_place_rotated", "multiview_rotated", "grid_lines"
if(dataset_category=="grid_lines"):
    num_angles = 180
else:
    num_angles = 180
if(dataset_category == "natural_cropped"):
    input_img_dir = "/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/data/filtered/natural/imagenet/rotated_cropped/1_degree_FIXED"
    dataset_list = ["dog","lizard","train","fish","indoor","beach"]
    common_query = True
    PROMPT = "How much is the second image rotated clockwise when compared to the first image?"
elif(dataset_category == "natural_cropped_gaussian_blur"): 
    input_img_dir = "/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/data/filtered/natural/imagenet/rotated_cropped_gaussian_blur/1_degree_FIXED"
    dataset_list = ["dog","fish"]
    common_query = True
    PROMPT = "How much is the second image rotated clockwise when compared to the first image?"
elif(dataset_category == "in_place_rotated"):
    # input_img_dir = "/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/data/filtered/natural/coco/1_degree_FIXED"
    # dataset_list = ["000000500634","000000532062"]
    # PROMPT_LIST = ["How much is the glass with water on the table rotated clockwise in degrees when compared to the first image?",
    #                 "How much is the milkcan, glass with water, and 2 glasses with milk on the table rotated clockwise in degrees when compared to the first image, assuming all the 4 objects are rotated by the same amounts?"]
    input_img_dir = "/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/data/filtered/natural/si-score/1_degree_FIXED"
    dataset_list = ["vase-indoor","vase-toaster-indoor"]#"koala-beach","vase-indoor","vase-toaster-indoor"
    common_query = False
    PROMPT_LIST = ["How much is the koala rotated clockwise in degrees when compared to the first image?",
                    "How much is the vase with green flowers rotated clockwise in degrees when compared to the first image?",
                    "How much is the vase with green flowers and the toaster rotated clockwise in degrees when compared to the first image, assuming the two objects are rotated by the same amounts?"]
elif(dataset_category == "multiview_rotated"):
    input_img_dir = "/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/data/filtered/natural/navi/1_degree_FIXED"
    dataset_list = ["garbage_truck_green_toy_s_002", "garbage_truck_green_toy_s_004"]
    common_query = True
    PROMPT = "How much is the garbage truck in the centre of the second image rotated clockwise when compared to the first image?"
elif(dataset_category == "grid_lines"):
    input_img_dir = "/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/data/filtered/grid_lines/1_degree_FIXED"
    # parent_dataset_list = ["dog","train","lizard","beach","indoor","fish","white"]
    # subdataset_list = ["horizontal","vertical","grid","checkerboard","checkerboard_dense","counter_clockwise_grid","curly_grid","static_dense_checkerboard"]
    # dataset_list = [
    # f"{parent}-{sub}"
    # for parent in parent_dataset_list
    # for sub in subdataset_list
    # ]
    parent_dataset_list = ["white"]

    subdataset_list = [
        "curly_grid"
    ]

    dataset_list = [
        f"{parent}-{sub}"
        for parent in parent_dataset_list
        for sub in subdataset_list
        if not (
            parent == "white" and 
            sub in ["counter_clockwise_grid", "static_dense_checkerboard"]
        )
    ]
    common_query = True
    PROMPT = "How much is the second image rotated clockwise when compared to the first?"

SAVE_PARENT_DIR = "output_embeddings"
output_response_parent_path = "/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/generate_files_for_llava/qs_list/natural/output_ans/llava-onevision-qwen2-7b-ov/what_angle"
os.makedirs(SAVE_PARENT_DIR, exist_ok=True)
if(dataset_category =="grid_lines"):
        new_category = f"{dataset_category}_{num_angles}"
        output_response_final_path = os.path.join(output_response_parent_path, new_category)
        os.makedirs(output_response_final_path, exist_ok=True)
else:
    output_response_final_path = os.path.join(output_response_parent_path, dataset_category)
    os.makedirs(output_response_final_path, exist_ok=True)

# ─── LOAD MODEL ONCE ──────────────────────────────────────
tokenizer, model, image_processor, max_length = load_pretrained_model(
    MODEL_PATH,
    None,
    "llava_qwen",
    device_map="auto",
    attn_implementation="sdpa"
)

model.eval()
device = model.device

for index, dataset in enumerate(dataset_list):

    print(f"\n\n=== Processing dataset: {dataset} ===")
    if(dataset_category =="grid_lines"):
        parent_dataset = dataset.split("-")[0]
        IMAGE_1 = os.path.join(input_img_dir, parent_dataset, dataset, "scale1", f"{dataset}_0.png")
        IMAGE_2_DIR = os.path.join(input_img_dir, parent_dataset, dataset, "scale1")
    else:
        IMAGE_1 = os.path.join(input_img_dir, dataset, "scale1", f"{dataset}_0.png")
        IMAGE_2_DIR = os.path.join(input_img_dir, dataset, "scale1")
    if(dataset_category =="grid_lines"):
        new_category = f"{dataset_category}_{num_angles}"
        SAVE_DIR = os.path.join(SAVE_PARENT_DIR, new_category, dataset,"scale1")
    else:
        SAVE_DIR = os.path.join(SAVE_PARENT_DIR, dataset_category, dataset,"scale1")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # ─── LOAD IMAGE 1 ONCE ────────────────────────────────────
    image1 = Image.open(IMAGE_1).convert("RGB")

    # ─── BUILD PROMPT ONCE ────────────────────────────────────
    if common_query:
        question = DEFAULT_IMAGE_TOKEN + "\n" + DEFAULT_IMAGE_TOKEN + "\n" + PROMPT
    else:
        PROMPT = PROMPT_LIST[index]
        question = DEFAULT_IMAGE_TOKEN + "\n" + DEFAULT_IMAGE_TOKEN + "\n" + PROMPT_LIST[index]

    conv = copy.deepcopy(conv_templates["qwen_1_5"])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    # ─── STORAGE DICTS ────────────────────────────────────────
    llm_hidden_dict = {}
    input_embeds_dict = {}
    vision_dict = {}
    proj_dict = {}
    responses_list = []  # <-- Store responses for JSONL

    # ─── LOOP OVER IMAGE_2 FILENAMES ──────────────────────────
    for i in range(num_angles):
        image2_path = os.path.join(IMAGE_2_DIR, f"{dataset}_{i}.png")
        print(f"Debug : Processing image: {image2_path}")

        image2 = Image.open(image2_path).convert("RGB")
        print(f"Debug : Loaded images. Image 1 size: {image1.size}, Image 2 size: {image2.size}")

        image_tensors = process_images([image1, image2], image_processor, model.config)
        print(f"Debug : Processed {len(image_tensors)} images. Shapes: {[img.shape for img in image_tensors]}")
        image_tensors = [img.to(dtype=torch.float16, device=device) for img in image_tensors]

        with torch.inference_mode():
            output_ids = model.generate(  # <-- Capture output_ids
                input_ids,
                images=image_tensors,
                image_sizes=[image1.size, image2.size],
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_new_tokens=32,
                save_name_for_embeddings="pair_test"
            )

        # ─── DECODE RESPONSE ───────────────────────────────────
        response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # ─── ADD TO RESPONSES LIST ────────────────────────────
        responses_list.append({
            "img_id": i,
            "prompt": PROMPT,
            "response": response,
            "model_id": MODEL_PATH
        })

        emb = embedding_outputs_global
        key = str(i)

        llm_hidden_dict[key] = emb["pair_test_llm_out_hidden"].detach().cpu().numpy()
        input_embeds_dict[key] = emb["pair_test_final_input_embeds"].detach().cpu().numpy()
        vision_dict[key] = emb["pair_test_vision_tower"].detach().cpu().numpy()
        proj_dict[key] = emb["pair_test_after_projector"].detach().cpu().numpy()

    if(save_result):

        # ─── SAVE EMBEDDINGS ───────────────────────────────────────
        np.save(os.path.join(SAVE_DIR, "llm_hidden.npy"), llm_hidden_dict)
        np.save(os.path.join(SAVE_DIR, "input_embeds.npy"), input_embeds_dict)
        np.save(os.path.join(SAVE_DIR, "vision_tower.npy"), vision_dict)
        np.save(os.path.join(SAVE_DIR, "after_projector.npy"), proj_dict)

        # ─── SAVE RESPONSES TO JSONL ───────────────────────────────
        jsonl_filename = f"{dataset}.jsonl"
        jsonl_filepath = os.path.join(output_response_final_path, jsonl_filename)
        
        with open(jsonl_filepath, 'w') as f:
            for response_obj in responses_list:
                f.write(json.dumps(response_obj) + '\n')

        print(f"✅ Saved embeddings for {dataset}")
        print(f"✅ Saved responses to {jsonl_filepath}")

    
    else:
        print(f"✅ Processed {dataset} - embeddings not saved as per configuration.")