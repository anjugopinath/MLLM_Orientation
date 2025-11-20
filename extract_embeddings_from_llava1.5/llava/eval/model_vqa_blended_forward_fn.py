import sys
sys.path.insert(0, "/<dirname>/<name>/g-t_embedding_classifier/llava/LLaVA")

import argparse

import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math
from plot_pca import *;
from llava.shared_data import embedding_outputs_global
print("checking imports : ",load_pretrained_model.__code__.co_filename)

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

# Function to save embeddings to a file
def save_embeddings_to_file(output_folder_name, str_increment, shape_name, blended_scale_subdir, subfolder_name, layername, embedding_outputs):

    #filename = f"embeddings_{filenames[index]}.npy"
    filename = f"{layername}.npy"
    parent_folder = os.path.join(output_folder_name, str_increment, shape_name, blended_scale_subdir, subfolder_name)
    os.makedirs(parent_folder, exist_ok=True)
    savepath = os.path.join(output_folder_name, str_increment, shape_name, blended_scale_subdir, subfolder_name, filename)
    np.save(savepath, embedding_outputs)
    print(f"Saved embeddings to {filename}")

def color_mapping_4_rotations(image_file):

    if("90" in image_file):
        print("90 degrees")
        color = "red"
    elif("180" in image_file):
        print("180 degrees")
        color = "orange"
    elif("270" in image_file):
        print("270 degrees")
        color = "green"
    elif("dog" in image_file):
        print("dog")
        color = "purple"

    # print("Image File : ",image_file, " color : ", color)

    return color

def color_mapping_72_rotations_45_interval(image_file):

    if("_" not in image_file):
        color = "black"
        angle = 365
    else:
        col_int = int(image_file.split("_")[-1].split(".")[0])
        angle = col_int
        if(col_int<=45):
            color = "green"
        elif(col_int<=90):
            color = "cyan"
        elif(col_int<=135):
            color = "blue"
        elif(col_int<=180):
            color = "purple"
        elif(col_int<=225):
            color = "magenta"
        elif(col_int<=270):
            color = "red"
        elif(col_int<=315):
            color = "yellow"
        elif(col_int<=360):
            color = "black"

    print("Image File : ",image_file, " color : ", color)
    return color, angle

def color_mapping_72_rotations_35_interval(image_file):

    if("_" not in image_file):
        color = "green"
    else:
        col_int = int(image_file.split("_")[-1].split(".")[0])
        if(col_int<=35):
            color = "green"
        elif(col_int<=75):
            color = "cyan"
        elif(col_int<=115):
            color = "blue"
        elif(col_int<=155):
            color = "purple"
        elif(col_int<=195):
            color = "magenta"
        elif(col_int<=235):
            color = "red"
        elif(col_int<=275):
            color = "yellow"
        elif(col_int<=315):
            color = "gray"
        elif(col_int<=355):
            color = "black"

    print("Image File : ",image_file, " color : ", color)
    return color


def print_layer_names(module, prefix=''):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.ModuleList):
            for i, layer in enumerate(child):
                print_layer_names(layer, f"{prefix}.{name}[{i}]")
        else:
            print(f"{prefix}.{name}")
        if isinstance(child, torch.nn.Module):
            print_layer_names(child, f"{prefix}.{name}")

def eval_model(args):

    count_pca_plot = 1
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    print("which one: ",load_pretrained_model.__code__.co_filename)

    print("Loaded pretrained model:", model_name)
    print("Model base is : ", args.model_base)
    print("Model path is : ", model_path)
    print("tokenizer is : ", tokenizer)
    # print("model is : ", model)
    # print("image_processor is : ", image_processor)
    print("context_len is : ", context_len)
    
    # print("model is : ", model)
    
    # print("Model layer names is : \n",)
    # print_layer_names(model)
    # sys.exit(1)

    # Assuming model is already loaded as described before
    # Register the hook to the CLIP encoder layer
    #clip_encoder_layer = model.model.vision_tower.vision_tower.vision_model.encoder
    # second_last_layer = model.model.layers[-2]
    # hook_clip = clip_encoder_layer.register_forward_hook(get_clip_encoder_output)  

    #sys.exit(1)
    
    #clip_encoder_outputs = []
    #second_last_layer_outputs = []

    # var_layers = ['lm_head']
    # filenames = ['lm_head']

    # var_layers = ['model.layers.39.post_attention_layernorm',
    # 'model.norm',
    # 'model.vision_tower.vision_tower.vision_model.encoder.layers.0.layer_norm2',
    # 'model.vision_tower.vision_tower.vision_model.encoder.layers.23.layer_norm2',
    # 'model.vision_tower.vision_tower.vision_model.post_layernorm',
    # 'model.mm_projector.0',
    # 'model.mm_projector.2',
    # 'lm_head']
    # filenames = ['layers39_post_attention_layernorm',
    # 'model_norm',
    # 'vision_tower_encoder0_layernorm2',
    # 'vision_tower_encoder23_layernorm2',
    # 'vision_tower_postlayernorm',
    # 'mm_projector0',
    # 'mm_projector2',
    # 'lm_head']

    # var_layers = ['model.vision_tower.vision_tower.vision_model.embeddings.patch_embedding',
	# 'model.vision_tower.vision_tower.vision_model.embeddings.position_embedding',
	# 'model.vision_tower.vision_tower.vision_model.pre_layrnorm',
	# 'model.vision_tower.vision_tower.vision_model.encoder.layers.0.self_attn.k_proj',
	# 'model.vision_tower.vision_tower.vision_model.encoder.layers.0.self_attn.v_proj',
	# 'model.vision_tower.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj',
	# 'model.vision_tower.vision_tower.vision_model.encoder.layers.0.self_attn.out_proj',
	# 'model.vision_tower.vision_tower.vision_model.encoder.layers.0.layer_norm1',
	# 'model.vision_tower.vision_tower.vision_model.encoder.layers.0.mlp.activation_fn',
	# 'model.vision_tower.vision_tower.vision_model.encoder.layers.0.mlp.fc1',
	# 'model.vision_tower.vision_tower.vision_model.encoder.layers.0.mlp.fc2',
	# 'model.vision_tower.vision_tower.vision_model.encoder.layers.0.layer_norm2',
	# 'model.vision_tower.vision_tower.vision_model.encoder.layers.23.self_attn.k_proj',
	# 'model.vision_tower.vision_tower.vision_model.encoder.layers.23.self_attn.v_proj',
	# 'model.vision_tower.vision_tower.vision_model.encoder.layers.23.self_attn.q_proj',
	# 'model.vision_tower.vision_tower.vision_model.encoder.layers.23.self_attn.out_proj',
	# 'model.vision_tower.vision_tower.vision_model.encoder.layers.23.layer_norm1',
	# 'model.vision_tower.vision_tower.vision_model.encoder.layers.23.mlp.activation_fn',
	# 'model.vision_tower.vision_tower.vision_model.encoder.layers.23.mlp.fc1',
	# 'model.vision_tower.vision_tower.vision_model.encoder.layers.23.mlp.fc2',
	# 'model.vision_tower.vision_tower.vision_model.encoder.layers.23.layer_norm2',
    # 'model.vision_tower.vision_tower.vision_model.post_layernorm']

    var_layers = ["model.vision_tower.vision_tower.vision_model.post_layernorm"]
    # var_layers = ['model.embed_tokens',
    # 'model.layers.0.post_attention_layernorm',
    # 'model.layers.39.post_attention_layernorm',
    # 'model.norm',
    # 'model.mm_projector.2',
    # 'lm_head']


    # filenames = ['vision_model.embeddings.patch_embedding',
    #     'vision_model.embeddings.position_embedding',
    #     'vision_model.pre_layrnorm',
    #     'vision_model.encoder.layers.0.self_attn.k_proj',
    #     'vision_model.encoder.layers.0.self_attn.v_proj',
    #     'vision_model.encoder.layers.0.self_attn.q_proj',
    #     'vision_model.encoder.layers.0.self_attn.out_proj',
    #     'vision_model.encoder.layers.0.layer_norm1',
    #     'vision_model.encoder.layers.0.mlp.activation_fn',
    #     'vision_model.encoder.layers.0.mlp.fc1',
    #     'vision_model.encoder.layers.0.mlp.fc2',
    #     'vision_model.encoder.layers.0.layer_norm2',
    #     'vision_model.encoder.layers.23.self_attn.k_proj',
    #     'vision_model.encoder.layers.23.self_attn.v_proj',
    #     'vision_model.encoder.layers.23.self_attn.q_proj',
    #     'vision_model.encoder.layers.23.self_attn.out_proj',
    #     'vision_model.encoder.layers.23.layer_norm1',
    #     'vision_model.encoder.layers.23.mlp.activation_fn',
    #     'vision_model.encoder.layers.23.mlp.fc1',
    #     'vision_model.encoder.layers.23.mlp.fc2',
    #     'vision_model.encoder.layers.23.layer_norm2',
    #     'vision_model.post_layernorm']

    filenames = ["vision_model.encoder.layers.23.layer_norm2"]

    # filenames = ['model.embed_tokens',
    # 'model.layers.0.post_attention_layernorm',
    # 'model.layers.39.post_attention_layernorm',
    # 'model.norm',
    # 'model.mm_projector.2',
    # 'lm_head']

    foreground_list = ['dog', 'lizard', 'train', 'rectangleTriangle']
    # background_list = ['fish', 'indoor', 'building', 'beach']
    # background_list = ["chessboard", "grid_lines", "horizontal_lines", "vertical_lines"]
    background_list = ["grid_lines", "horizontal_lines", "vertical_lines"]

    str_increment = "1_degree_FIXED" #"1_degree", "5_degrees"
    counter_rot = False
    synthetic_bg = True
    if(synthetic_bg):
        required_shape_list = ["dog"]#["dog","lizard", "train"]
    else:
        required_shape_list = ["lizard_on_fish","train_on_indoor"]#["dog_on_beach","lizard_on_fish","train_on_indoor"]
    # required_shape = "train_on_indoor"
    if(counter_rot):
        counter_rot_names = ["_b.5Rot_f.5Rot_1degreeTotal","_bRot_fStat"] #"_bRot_fStat", "_b.5Rot_f.5Rot_1degreeTotal"
    else:
        counter_rot_names = [""] #"_bRot_fStat", "_b.5Rot_f.5Rot_1degreeTotal"
    int_increment = 1 #1, 5
    # input_images_blended_path = f"/<dirname>/<name>/g-t_embedding_classifier/data/filtered/composition/composed_rotated/{str_increment}"
    input_qs_blended_path = f"/<dirname>/<name>/g-t_embedding_classifier/generate_files_for_llava/qs_list/blended/input_qs/what_angle/{str_increment}"
    output_ans_blended_path = f"/<dirname>/<name>/g-t_embedding_classifier/generate_files_for_llava/qs_list/blended/output_ans/llava-v1.5-13b/what_angle/{str_increment}"
    image_folder = f"/<dirname>/<name>/g-t_embedding_classifier/data/filtered/composition/composed_rotated/{str_increment}"

    for foreground in foreground_list:
        for background in background_list:
            for counter_rot_name in counter_rot_names:
                if(synthetic_bg):
                    blended_shape = foreground + "_on_" + background
                    # if("dog" not in blended_shape):
                    if not any(x in blended_shape for x in required_shape_list):
                        continue
                    if(counter_rot):
                        blended_shape = foreground + "_on_" + background + f"{counter_rot_name}"
                else:
                    blended_shape = foreground + "_on_" + background
                    if(blended_shape not in required_shape_list):
                        continue
                    if(counter_rot):
                        blended_shape = foreground + "_on_" + background + f"{counter_rot_name}"
                blended_shape_orig = foreground + "_on_" + background
                print("Blended shape is : ", blended_shape)
                print("Blended shape original is : ", blended_shape_orig)
                
                print("Processing shape: ", blended_shape)
                for scaleIndex in range(60):
                    print("Scale index is : ", scaleIndex)
                    # if scaleIndex not in [0,1,3,7,15,31,59]:
                    # if scaleIndex not in [0,3,15]:
                    if scaleIndex not in [3,15]:
                    # if scaleIndex not in [0]:
                    # if scaleIndex not in [3,4,5]:
                        continue
                    print("Scale is : ", scaleIndex)
                    blended_scale_subdir = "scale"+str(scaleIndex+1)

                    for index, entry in enumerate(var_layers):
                        print("Layer is : ", entry)
                        var_layer = model.get_submodule(entry)
                        #hook_var = var_layer.register_forward_hook(get_var_layer_output)
                        #var_layer_outputs = []
                        #collected_embeddings = []
                        colors = []
                        angles = []
                        # Define a global variable to store all embeddings
                        # embedding_outputs = {}
                        
                        # question_file = os.path.join(input_qs_blended_path, blended_shape, f"{blended_shape}_blended_360_{int_increment}_degrees.jsonl")
                        question_file = os.path.join(input_qs_blended_path, blended_shape_orig, f"{blended_shape_orig}_blended_360_{int_increment}_degrees.jsonl")
                        questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
                        questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

                        answers_file = os.path.join(output_ans_blended_path, blended_shape, blended_scale_subdir, f"{blended_shape}_blended_360_{int_increment}_degrees.jsonl")
                        os.makedirs(os.path.join(output_ans_blended_path, blended_shape, blended_scale_subdir), exist_ok=True)
                        answers_file = os.path.expanduser(answers_file)
                        ans_file = open(answers_file, "w")
                        print("size of questions : ", len(questions))
                        for line in tqdm(questions):
                            idx = line["question_id"]
                            image_file = line["image"]
                            # print("image_file : ", image_file)
                            qs = line["text"]
                            cur_prompt = qs
                            if model.config.mm_use_im_start_end:
                                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
                            else:
                                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

                            conv = conv_templates[args.conv_mode].copy()
                            conv.append_message(conv.roles[0], qs)
                            conv.append_message(conv.roles[1], None)
                            prompt = conv.get_prompt()

                            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

                            image = Image.open(os.path.join(image_folder, blended_shape, blended_scale_subdir, image_file))
                            print("Full image path : ", os.path.join(image_folder, blended_shape, blended_scale_subdir, image_file) )
                            print("\nimage path : ", image_file)
                            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                            print("After calling image_processor, image_tensor shape is : ", image_tensor.shape)
                            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                            keywords = [stop_str]
                            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
                            # print("model inside eval_model : ", model)
                            save_name = image_file.split("_")[-1].replace(".png", "")
                            print("save_name : ", save_name)
                            # if(int(save_name)>10):
                            #     break
                            print("Before calling torch.inference_mode")
                            with torch.inference_mode():
                                output_ids = model.generate(
                                    input_ids,
                                    images=image_tensor.unsqueeze(0).half().cuda(),
                                    do_sample=True if args.temperature > 0 else False,
                                    temperature=args.temperature,
                                    top_p=args.top_p,
                                    num_beams=args.num_beams,
                                    # no_repeat_ngram_size=3,
                                    max_new_tokens=1024,
                                    use_cache=True,
                                    save_name_for_embeddings=save_name)
                            print("After calling torch.inference_mode")

                            vision_model_debug = model.get_vision_tower().vision_tower.vision_model

                            with torch.no_grad():
                                embeddings_debug = vision_model_debug.embeddings(image_tensor.unsqueeze(0).half().cuda())  # patch + pos embedding
                                output_debug_2 = vision_model_debug.encoder.layers[-2](embeddings_debug,attention_mask=None,causal_attention_mask=None)

                            print("Output -2 Type:", type(output_debug_2))
                            print("Shape of Output -2:", output_debug_2[0].shape)

                            with torch.no_grad():
                                embeddings_debug = vision_model_debug.embeddings(image_tensor.unsqueeze(0).half().cuda())  # patch + pos embedding
                                output_debug = vision_model_debug.encoder.layers[-1](embeddings_debug,attention_mask=None,causal_attention_mask=None)

                            print("Output Type:", type(output_debug))
                            print("Shape of Output :", output_debug[0].shape)

                            with torch.no_grad():
                                embeddings_debug = vision_model_debug.embeddings(image_tensor.unsqueeze(0).half().cuda())  # patch + pos embedding
                                output_debug = vision_model_debug.post_layernorm(embeddings_debug)

                            print("Post Layernorm Type:", type(output_debug))
                            print("Shape of Post Layernorm :", output_debug[0].shape)

                            # for name, module in model.named_modules():
                            #     if 'projector' in name:
                            #         print(f"Found projector module: {name} -> {module}")
                            with torch.no_grad():
                                output = vision_model_debug(image_tensor.unsqueeze(0).half().cuda(), output_hidden_states=True)

                                # If your config uses the second-last hidden state (e.g., layer -2)
                                image_features = output.hidden_states[-2][:, 1:, :]  # remove CLS token if using 'patch' mode

                                # Now pass the tensor to the projector
                                projected_features = model.model.mm_projector(image_features)


                            print("Type of projected features:", type(projected_features))
                            print("Shape of projected features:", projected_features.shape)



                            # print("Output Type:", type(output_debug))
                            # print("Shape of Output:", output_debug[0].shape)
                            # for i, elem in enumerate(output_debug):
                            #     print(f"Element {i} type: {type(elem)}")
                            #     if hasattr(elem, "shape"):
                            #         print(f"Element {i} shape: {elem.shape}")

                            # hook_var.remove() # Remove the hook immediately after generation
                            # embedding = var_layer_output.cpu().numpy()
                            # save_name = image_file.split("_")[-1].replace(".png", "")
                            # print("save_name : ", save_name)
                            # embedding_outputs[save_name] = embedding

                            input_token_len = input_ids.shape[1]
                            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                            if n_diff_input_output > 0:
                                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
                            outputs = outputs.strip()
                            if outputs.endswith(stop_str):
                                outputs = outputs[:-len(stop_str)]
                            outputs = outputs.strip()

                            ans_id = shortuuid.uuid()
                            ans_file.write(json.dumps({"question_id": idx,
                                                    "prompt": cur_prompt,
                                                    "text": outputs,
                                                    "answer_id": ans_id,
                                                    "model_id": model_name,
                                                    "metadata": {}}) + "\n")
                            ans_file.flush()
                            #f = open("hook_output.txt","a")
                            #f.write(str(clip_encoder_output))
                            #shape_text = "shape is : " + str(clip_encoder_output.shape)
                            #f.write(shape_text)
                            #color = color_mapping_4_rotations(image_file)
                            color, angle = color_mapping_72_rotations_45_interval(image_file)
                            #clip_encoder_outputs.append(clip_encoder_output)
                            #second_last_layer_outputs. append(second_last_layer_output)
                            #var_layer_outputs.append(var_layer_output)
                            #embedding = var_layer_output.cpu().numpy()  # Move to CPU and convert to numpy
                            # save_name = image_file.split("_")[-1].split(".")[0]
                            # save_name = image_file.split("_")[-1].replace(".png", "")
                            # print("save_name : ", save_name)
                            # embedding_outputs[save_name] = embedding
                            colors.append(color)
                            angles.append(angle)
                            count_pca_plot +=1
                            
                        qs_type = "what_angle" #["what_angle", "what_is_color"]
                        output_folder_name = f"pca_images/{qs_type}"
                        shape_name = blended_shape #"sampleQs_bird_VisionEnc_1_degree" #Folder name in pca_images folder

                        imgname = filenames[index] + ".png"
                        

                        subfolder_name = "embeddings"
                        layername = filenames[index]
                        save_embeddings_to_file(output_folder_name, str_increment, shape_name, blended_scale_subdir, subfolder_name, layername, embedding_outputs_global)
                    
    # ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--image-folder", type=str, default="")
    # parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    # parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
