import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------
# Using embeddings 3 and 15, replace values in 15 with those from 3
# at locations with greatest absolute differences, in descending order.
# Do this for different number of locations (step size until max).
#------------------------------------------------------------------------
# other_nums = [2, 15, 39, 75, 111, 147, 183, 219, 255, 291, 327]
# other_nums = [5, 15, 39, 72, 108, 147, 180, 220, 248, 286, 324]

modelnames = ["Qwen2.5-VL-7B-Instruct"] #llava-v1.5-13b, llava-v1.6-vicuna-13b, "llava-ov-qwen2-7b", "Qwen2.5-VL-7B-Instruct"
origScales = ["scale1"] #["scale1", "scale4", "scale16"]
category = "in_place_rotated" #blended | natural_cropped | in_place_rotated
if(category=="natural_cropped"):
    blended_shapes = ["dog","train","lizard","beach","fish","indoor"]
elif(category=="in_place_rotated"):
    blended_shapes = ["koala-beach","vase-indoor","vase-toaster-indoor"]

# blended_shapes = ["lizard_on_fish", "train_on_indoor"]#dog_on_beach, lizard_on_fish, train_on_indoor
angle_str_plural = "1_degrees_FIXED" #"1_degrees", "1_degrees_FIXED"
angle_str_singular = "1_degree_FIXED" #"1_degree", "1_degree_FIXED"

for modelname in modelnames:
    print(f"\n================= Processing model: {modelname} =================")
    for blended_shape in blended_shapes:    
        print(f"\n********** Processing blended shape: {blended_shape} **********")
        # Define total bins and step depending on model
        if modelname == "llava-v1.5-13b":
            other_nums = [5, 15, 72, 147, 220, 286, 324]
            anchor = 3
            num_bins_total = 589824
            # step = 20000
            # step = 60000
            step = [15000, 30000, 60000, 120000, 180000, 240000, 300000, 360000, 420000, 480000]
        elif modelname == "llava-v1.6-vicuna-13b":
            other_nums = [5, 15, 72, 147, 220, 286, 324]
            anchor = 3
            if(blended_shape == "dog_on_beach"):
                num_bins_total = 2949120
                step = [8000, 16000, 128000, 270000, 540000, 810000, 1080000, 1620000, 1890000, 2160000]
            elif(blended_shape == "lizard_on_fish" or blended_shape == "train_on_indoor"):
                num_bins_total = 1769472
                step = [8000, 16000, 128000, 270000, 540000, 810000, 1080000, 1620000]
        elif(modelname == "llava-ov-qwen2-7b"):
            other_nums = [15, 30, 78, 118, 148, 173]
            anchor = 9
            if(category=="natural_cropped" or category=="in_place_rotated"):
                num_bins_total = 3359232
                step = [8000, 16000, 128000, 270000, 540000, 810000, 1080000, 1620000, 1890000, 2160000, 3000000]

        elif(modelname == "Qwen2.5-VL-7B-Instruct"):
            other_nums = [15, 30, 78, 118, 148, 173]
            anchor = 9
            if(category=="natural_cropped" or category=="in_place_rotated"):
                if(blended_shape in ["fish","indoor","train"]):
                    num_bins_total = 501760
                    step = [2000, 5000, 8000, 16000, 25000, 75000, 128000, 270000, 335000, 500000]
                else:
                    num_bins_total = 829440
                    step = [2000, 8000, 16000, 25000, 75000, 128000, 270000, 335000, 500000, 800000]

        # for num_bins in range(step, num_bins_total + 1, step):
        for num_bins in step:
            print(f"num_bins is : {num_bins}")
            for scale in origScales:
                print(f"\n********** Processing scale: {scale} **********")
                for num in other_nums:
                    # pair_str = f"{anchor}into{num}_{num_bins}countbins"
                    pair_str = f"{anchor}into{num}"

                    if modelname == "llava-v1.5-13b":
                        ref_emb = f"/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/llava/LLaVA/llava/eval/pca_images/what_angle/{angle_str_singular}/{blended_shape}/{scale}/embeddings/vision_model.encoder.layers.23.layer_norm2.npy"
                        out_emb_dir = f"/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/llava/LLaVA/llava/eval/pca_images/what_angle/{angle_str_singular}/{blended_shape}/{scale}_{pair_str}_{num_bins}absdiff/embeddings"

                    elif modelname == "llava-v1.6-vicuna-13b":
                        ref_emb = f"/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/llava1.6/LLaVA/llava/eval/pca_images/what_angle/{angle_str_singular}/{blended_shape}/{scale}/embeddings/vision_model.encoder.layers.23.layer_norm2.npy"
                        out_emb_dir = f"/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/llava1.6/LLaVA/llava/eval/pca_images/what_angle/{angle_str_singular}/{blended_shape}/{scale}_{pair_str}_{num_bins}absdiff/embeddings"

                    elif(modelname == "llava-ov-qwen2-7b"):
                        #Reference embedding file
                        ref_emb = f"/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/llava_next/LLaVA-NeXT/llava/eval/output_embeddings/{category}/{blended_shape}/{scale}/vision_tower.npy"
                        #Out dirs for new embeddings
                        out_emb_dir = f"/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/llava_next/LLaVA-NeXT/llava/eval/output_embeddings/{category}/{blended_shape}/{scale}_{pair_str}_{num_bins}absdiff"
                    elif(modelname == "Qwen2.5-VL-7B-Instruct"):
                        #Reference embedding file
                        ref_emb = f"/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/Qwen2.5-VL-7B/github/transformers/src/transformers/models/qwen2_5_vl/output_embeddings/{category}/{blended_shape}/{scale}/vision_tower.npy"
                        #Out dirs for new embeddings
                        out_emb_dir = f"/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/Qwen2.5-VL-7B/github/transformers/src/transformers/models/qwen2_5_vl/output_embeddings/{category}/{blended_shape}/{scale}_{pair_str}_{num_bins}absdiff"
                    os.makedirs(out_emb_dir, exist_ok=True)

                    #------------------------------------------------------------
                    # Load embeddings
                    #------------------------------------------------------------
                    embedding = np.load(ref_emb, allow_pickle=True).item()
                    # embedding_copy = {k: v.clone() for k, v in embedding.items()}
                    if(modelname == "llava-ov-qwen2-7b" or modelname == "Qwen2.5-VL-7B-Instruct"):
                        embedding_copy = {k: v.copy() for k, v in embedding.items()}
                    else:
                        embedding_copy = {k: v.clone() for k, v in embedding.items()}

                    #------------------------------------------------------------
                    # Compute absolute differences between embeddings 3 and 15
                    #------------------------------------------------------------
                    if modelname == "llava-v1.5-13b":
                        # shape (1, 576, 1024)
                        diff = np.abs(embedding[str(num)][0]) - np.abs(embedding[str(anchor)][0])
                        abs_diff = np.abs(diff)

                        flat_diff = abs_diff.flatten()
                        top_indices = np.argsort(-flat_diff)[:num_bins]  # descending

                        # Replace
                        for idx in top_indices:
                            row = idx // 1024
                            col = idx % 1024
                            embedding_copy[str(num)][0, row, col] = embedding[str(anchor)][0, row, col]

                    elif (modelname == "llava-v1.6-vicuna-13b" or modelname == "llava-ov-qwen2-7b"):
                        # shape (5, 576, 1024)
                        diff = np.abs(embedding[str(anchor)]) - np.abs(embedding[str(num)])
                        abs_diff = np.abs(diff)

                        flat_diff = abs_diff.flatten()
                        top_indices = np.argsort(-flat_diff)[:num_bins]

                        num_rows, num_cols = embedding[str(anchor)].shape[1], embedding[str(anchor)].shape[2]

                        for idx in top_indices:
                            batch = idx // (num_rows * num_cols)
                            rem = idx % (num_rows * num_cols)
                            row = rem // num_cols
                            col = rem % num_cols
                            embedding_copy[str(num)][batch, row, col] = embedding[str(anchor)][batch, row, col]

                    elif modelname == "Qwen2.5-VL-7B-Instruct":
                        # shape (rows, 1280) e.g. (648,1280) or (392,1280)

                        emb_anchor = embedding[str(anchor)]
                        emb_num = embedding[str(num)]

                        # If flattened, reshape back
                        if emb_anchor.ndim == 1:
                            hidden = 1280
                            rows = emb_anchor.shape[0] // hidden
                            emb_anchor = emb_anchor.reshape(rows, hidden)
                            emb_num = emb_num.reshape(rows, hidden)

                        diff = np.abs(emb_anchor) - np.abs(emb_num)
                        abs_diff = np.abs(diff)

                        flat_diff = abs_diff.flatten()
                        top_indices = np.argsort(-flat_diff)[:num_bins]

                        num_rows, num_cols = emb_anchor.shape

                        for idx in top_indices:
                            row = idx // num_cols
                            col = idx % num_cols

                            embedding_copy[str(num)][row, col] = emb_anchor[row, col]

                    #------------------------------------------------------------
                    # Plot histogram of differences (once per run per scale)
                    #------------------------------------------------------------
                    # hist_path = os.path.join(out_emb_dir, "embedding_diff_histogram.png")
                    # if not os.path.exists(hist_path):
                    #     plt.figure(figsize=(7, 5))
                    #     plt.hist(flat_diff, bins=100, edgecolor="black")
                    #     plt.title("Histogram - Absolute Differences |embedding3 - embedding15|")
                    #     plt.xlabel("Absolute Difference")
                    #     plt.ylabel("Count")
                    #     plt.tight_layout()
                    #     plt.savefig(hist_path)
                    #     plt.close()
                    #     print(f"Saved histogram of differences at {hist_path}")

                    #------------------------------------------------------------
                    # Save modified embedding
                    #------------------------------------------------------------
                    if(modelname == "llava-ov-qwen2-7b" or modelname == "Qwen2.5-VL-7B-Instruct"):
                        new_emb_name = "vision_tower.npy"
                    else:
                        new_emb_name = "vision_model.encoder.layers.23.layer_norm2.npy"
                    final_emb_path = os.path.join(out_emb_dir, new_emb_name)
                    np.save(final_emb_path, embedding_copy)

                    print(f"Modified embedding saved at {final_emb_path}")
                    print(f"Completed processing for num_bins : {num_bins}")
