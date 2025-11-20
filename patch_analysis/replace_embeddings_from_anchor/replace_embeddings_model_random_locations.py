import os
import sys
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#---------------------------
# Create new embeddings by copying values from
#embedding 3 to embedding 15 at the indices corresponding to the top N weights by absolute value
# this is done for different number of locations (20k to max in steps of 20k)

# The locations to replace are picked entirely randomly from the top N weights

#-----------------------------
# other_nums = [2, 15, 39, 75, 111, 147, 183, 219, 255, 291, 327]
# other_nums = [5, 15, 39, 72, 108, 147, 180, 220, 248, 286, 324]
other_nums = [5, 15, 72, 147, 220, 286, 324]
anchor = 3
modelnames = ["llava-v1.5-13b","llava-v1.6-vicuna-13b"] #llava-v1.5-13b, llava-v1.6-vicuna-13b
origScales = ["scale1","scale4", "scale16"] #["scale1", "scale4", "scale16"]
# destScales = ["scale1"]
category = "blended"
blended_shapes = ["lizard_on_fish", "train_on_indoor"] #dog_on_beach, lizard_on_fish, train_on_indoor
angle_str_plural = "1_degrees_FIXED" #"1_degrees", "1_degrees_FIXED"
angle_str_singular = "1_degree_FIXED" #"1_degree", "1_degree_FIXED"
# Replace with your actual file paths
# if(modelname == "llava-v1.5-13b"):
#     file_sin1 = "/<dirname>/<name>/affine_tranformation/objects/DINO/regressor/Linear_Regression/output/LLAVA/llava-v1.5-13b/blended/dog_on_beach/1_degrees/scale1/_visionEncAll/clip_vitl336/what_angle/72_test_samples/Scaled/RidgeRegression/train_and_test"
# file_sin = "/<dirname>/<name>/affine_tranformation/objects/DINO/regressor/Linear_Regression/output/LLAVA/blended/dog_on_beach/1_degrees/scale1_3into15_30countbins/_visionEncAll/clip_vitl336/what_angle/72_test_samples/Scaled/RidgeRegression/train_and_test/sin_model_weights.csv"
# file_cos = "/<dirname>/<name>/affine_tranformation/objects/DINO/regressor/Linear_Regression/output/LLAVA/blended/dog_on_beach/1_degrees/scale1_3into15_30countbins/_visionEncAll/clip_vitl336/what_angle/72_test_samples/Scaled/RidgeRegression/train_and_test/cos_model_weights.csv"
# out_logs_dir = "/<dirname>/<name>/g-t_embedding_classifier/utils/logs"
# num_bins_total = 589824 #100, 2000, 200000, 471860, 589824
for modelname in modelnames:
    print(f"\n================= Processing model: {modelname} =================")
    for blended_shape in blended_shapes:    
        print(f"\n********** Processing blended shape: {blended_shape} **********")
        if(modelname == "llava-v1.5-13b"):
            num_bins_total = 589824
            # step = 20000
            # step = 60000
            step = [15000, 30000, 60000, 120000, 180000, 240000, 300000, 360000, 420000, 480000]
        elif(modelname == "llava-v1.6-vicuna-13b"):
            if(blended_shape == "dog_on_beach"):
                num_bins_total = 2949120
                step = [8000, 16000, 128000, 270000, 540000, 810000, 1080000, 1620000, 1890000, 2160000]
            elif(blended_shape == "lizard_on_fish" or blended_shape == "train_on_indoor"):
                num_bins_total = 1769472
                step = [8000, 16000, 128000, 270000, 540000, 810000, 1080000, 1620000]

        # for num_bins in range(step, num_bins_total + 1, step):
        for num_bins in step:
            # print(i)
            print(f"num_bins is : {num_bins}")
            for scale in origScales:
                # scale = "scale1" 
                print(f"\n********** Processing scale: {scale} **********")
                for num in other_nums:
                    # pair_str = f"{anchor}into{num}_{num_bins}countbins"
                    pair_str = f"{anchor}into{num}"
                    if(modelname == "llava-v1.5-13b"):
                        #Model weight files
                        # input_modelweights_folder = f"/<dirname>/<name>/affine_tranformation/objects/DINO/regressor/Linear_Regression/output/llava-v1.5-13b/{category}/{blended_shape}/{angle_str_plural}/{scale}/_visionEncAll/clip_vitl336/what_angle/72_test_samples/Scaled/RidgeRegression/train_and_test"
                        # input_file_sin = f"{input_modelweights_folder}/sin_model_weights.csv"
                        # input_file_cos = f"{input_modelweights_folder}/cos_model_weights.csv"
                        #Reference embedding file
                        ref_emb = f"/<dirname>/<name>/g-t_embedding_classifier/llava/LLaVA/llava/eval/pca_images/what_angle/{angle_str_singular}/{blended_shape}/{scale}/embeddings/vision_model.encoder.layers.23.layer_norm2.npy"
                        #Out dirs for new embeddings
                        out_emb_dir = f"/<dirname>/<name>/g-t_embedding_classifier/llava/LLaVA/llava/eval/pca_images/what_angle/{angle_str_singular}/{blended_shape}/{scale}_{pair_str}_{num_bins}random/embeddings"
                        # out_origlogs_dir = f"/<dirname>/<name>/affine_tranformation/objects/DINO/regressor/Linear_Regression/output/llava-v1.5-13b/{category}/{blended_shape}/{angle_str_plural}/{scale}/_visionEncAll/clip_vitl336/what_angle/72_test_samples/Scaled/RidgeRegression/train_and_test"
                    elif(modelname == "llava-v1.6-vicuna-13b"):
                        #Model weight files
                        # input_modelweights_folder = f"/<dirname>/<name>/affine_tranformation/objects/DINO/regressor/Linear_Regression/output/llava-v1.6-vicuna-13b/{category}/{blended_shape}/{angle_str_plural}/{scale}/_visionEncAll/clip_vitl336/what_angle/72_test_samples/Scaled/RidgeRegression/train_and_test"
                        # input_file_sin = f"{input_modelweights_folder}/sin_model_weights.csv"
                        # input_file_cos = f"{input_modelweights_folder}/cos_model_weights.csv"
                        #Reference embedding file
                        ref_emb = f"/<dirname>/<name>/g-t_embedding_classifier/llava1.6/LLaVA/llava/eval/pca_images/what_angle/{angle_str_singular}/{blended_shape}/{scale}/embeddings/vision_model.encoder.layers.23.layer_norm2.npy"
                        #Out dirs for new embeddings
                        out_emb_dir = f"/<dirname>/<name>/g-t_embedding_classifier/llava1.6/LLaVA/llava/eval/pca_images/what_angle/{angle_str_singular}/{blended_shape}/{scale}_{pair_str}_{num_bins}random/embeddings"
                        # out_logs_dir = "/<dirname>/<name>/affine_tranformation/objects/DINO/regressor/Linear_Regression/output/LLAVA/llava-v1.6-vicuna-13b/logs"
                    os.makedirs(out_emb_dir, exist_ok=True)

                    pd.set_option('display.float_format', lambda x: '%.20f' % x)

                    #----------------------------------------------------------------
                    #COPY VALUES FROM LOCATIONS IN EMBEDDING 3 TO EMBEDDING 15
                    #----------------------------------------------------------------
                    # Load embedding
                    embedding = np.load(ref_emb, allow_pickle=True).item()
                    # print("keys : ", embedding.keys())
                    # sys.exit(1)
                    # print("Type:", type(embedding))
                    # print("Shape:", getattr(embedding['0'], "shape", None))
                    # print("Dtype:", getattr(embedding, "dtype", None))
                    # sys.exit(1)

                    # Step 1: Make a copy
                    # embedding_copy = embedding.copy()
                    # Step 1: Make a copy of dict
                    # embedding_copy = {k: v.copy() for k, v in embedding.items()}
                    embedding_copy = {k: v.clone() for k, v in embedding.items()}
                    # print("embedding copy made : ", embedding_copy.keys())
                    # sys.exit(1)

                    seed = 42              # fixed seed for reproducibility
                    all_indices = list(range(num_bins_total))  # total locations available

                    random.seed(seed)  # set seed
                    target_indices = random.sample(all_indices, num_bins)
                    # target_indices = list(set(sin_indices + cos_indices))
                    # for k, v in embedding.items():
                    #     print(f"Key {k}: shape={v.shape}, dtype={v.dtype}")
                    #     break  # just show one for now
                    # sys.exit(1)

                    # Step 3: Replace values in row 14 with values from row 2 at those indices
                    if(modelname == "llava-v1.5-13b"):
                        for idx in target_indices:
                            row = idx // 1024
                            col = idx % 1024
                            embedding_copy[str(num)][0, row, col] = embedding[str(anchor)][0, row, col]
                    elif(modelname == "llava-v1.6-vicuna-13b"):
                    
                        num_rows, num_cols = embedding[str(num)].shape[1], embedding[str(anchor)].shape[2]  # (576, 1024)
                        print("num_rows, num_cols : ", num_rows, num_cols)
                        for idx in target_indices:
                            # decode flattened index (batch, row, col)
                            batch = idx // (num_rows * num_cols)
                            rem = idx % (num_rows * num_cols)
                            row = rem // num_cols
                            col = rem % num_cols

                            print(f"Replacing at index: {idx} -> (batch: {batch}, row: {row}, col: {col})")
                            # sys.exit(1)
                            embedding_copy[str(num)][batch, row, col] = embedding[str(anchor)][batch, row, col]

                    # Step 4: Save as new .npy file
                    # new_emb_path = ref_emb.replace(".npy", "_modified.npy")
                    new_emb_name = "vision_model.encoder.layers.23.layer_norm2.npy"
                    final_emb_path = os.path.join(out_emb_dir, new_emb_name)
                    print("Final emb path : ", final_emb_path)
                    np.save(final_emb_path, embedding_copy)

                    print(f"Modified embedding saved at {final_emb_path}")
                    print(f"Completed processing for num_bins : {num_bins}")