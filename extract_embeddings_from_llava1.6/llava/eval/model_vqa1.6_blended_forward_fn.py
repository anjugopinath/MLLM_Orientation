import argparse
import sys
import torch
import os
import json
import numpy as np
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math
from llava.shared_data import embedding_outputs_global

print("PRINTING TORCH AND CUDA VERSIONS:")
import torch
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.__version__)
# sys.exit(1)

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


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    print("Model is : ", model)
    # import torch
    # print("PRINTING MODEL AND TORCH VERSIONS:")
    # print(torch.version.cuda)
    # print(torch.backends.cudnn.version())

    # questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    # answers_file = os.path.expanduser(args.answers_file)
    # os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    # ans_file = open(answers_file, "w")
    #----------- INSERT NEW CODE HERE -----------

    foreground_list = ['dog', 'lizard', 'train', 'rectangleTriangle']
    # background_list = ['fish', 'indoor', 'building', 'beach']
    background_list = ["chessboard", "grid_lines", "horizontal_lines", "vertical_lines"]

    str_increment = "1_degree_FIXED" #"1_degree", "5_degrees"
    counter_rot = True
    synthetic_bg = True
    
    if(synthetic_bg):
        required_shape_list = ["train"]#["dog","lizard", "train"]
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
    output_ans_blended_path = f"/<dirname>/<name>/g-t_embedding_classifier/generate_files_for_llava/qs_list/blended/output_ans/llava-v1.6-vicuna-13b/what_angle/{str_increment}"
    image_folder = f"/<dirname>/<name>/g-t_embedding_classifier/data/filtered/composition/composed_rotated/{str_increment}"

    # var_layers = ["model.vision_tower.vision_tower.vision_model.post_layernorm"]
    var_layers = ["model.vision_tower.vision_tower.vision_model.encoder.layers.23.layer_norm2"]
    filenames = ["vision_model.encoder.layers.23.layer_norm2"]

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
                    #     continue
                    # if scaleIndex not in [0,3,15]:
                    if scaleIndex not in [0]:
                    # if scaleIndex not in [3,4,5]:
                        continue
                    print("Scale is : ", scaleIndex)
                    blended_scale_subdir = "scale"+str(scaleIndex+1)

                    for index, entry in enumerate(var_layers):
                        print("Layer is : ", entry)
                        var_layer = model.get_submodule(entry)

                        question_file = os.path.join(input_qs_blended_path, blended_shape_orig, f"{blended_shape_orig}_blended_360_{int_increment}_degrees.jsonl")
                        questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
                        questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

                        answers_file = os.path.join(output_ans_blended_path, blended_shape, blended_scale_subdir,f"{blended_shape}_blended_360_{int_increment}_degrees.jsonl")
                        os.makedirs(os.path.join(output_ans_blended_path, blended_shape, blended_scale_subdir), exist_ok=True)
                        answers_file = os.path.expanduser(answers_file)
                        ans_file = open(answers_file, "w")
                        print("size of questions : ", len(questions))  

        #------------ END OF NEW CODE HERE -----------

                        for line in tqdm(questions):
                            idx = line["question_id"]
                            image_file = line["image"]
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

                            # image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
                            image = Image.open(os.path.join(image_folder, blended_shape, blended_scale_subdir, image_file))
                            image_tensor = process_images([image], image_processor, model.config)[0]
                            save_name = image_file.split("_")[-1].replace(".png", "")
                            with torch.inference_mode():
                                output_ids = model.generate(
                                    input_ids,
                                    images=image_tensor.unsqueeze(0).half().cuda(),
                                    image_sizes=[image.size],
                                    do_sample=True if args.temperature > 0 else False,
                                    temperature=args.temperature,
                                    top_p=args.top_p,
                                    num_beams=args.num_beams,
                                    # no_repeat_ngram_size=3,
                                    max_new_tokens=1024,
                                    use_cache=True,
                                    save_name_for_embeddings=save_name)

                            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

                            ans_id = shortuuid.uuid()
                            ans_file.write(json.dumps({"question_id": idx,
                                                    "prompt": cur_prompt,
                                                    "text": outputs,
                                                    "answer_id": ans_id,
                                                    "model_id": model_name,
                                                    "metadata": {}}) + "\n")
                            ans_file.flush()
                            qs_type = "what_angle" #["what_angle", "what_is_color"]
                        output_folder_name = f"pca_images/{qs_type}"
                        shape_name = blended_shape #"sampleQs_bird_VisionEnc_1_degree" #Folder name in pca_images folder
                        imgname = filenames[index] + ".png"
                        subfolder_name = "embeddings"
                        layername = filenames[index]
                        save_embeddings_to_file(output_folder_name, str_increment, shape_name, blended_scale_subdir, subfolder_name, layername, embedding_outputs_global)
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
