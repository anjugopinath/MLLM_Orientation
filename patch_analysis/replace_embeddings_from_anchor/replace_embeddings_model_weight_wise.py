import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#---------------------------
# Using the sine and cosine model weights, create new embeddings by copying values from
#embedding 3 to embedding 15 at the indices corresponding to the top N weights by absolute value
# this is done for different number of locations (20k to max in steps of 20k)

#-----------------------------

# for num in other_nums:
#     pair_str = f"{anchor}into{num}"
#     print("pair str : ", pair_str)
# sys.exit(1)
# for num in other_nums:
#     print("diff : ", abs(num - anchor))
# sys.exit(1)
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
            # step = 90000
            # step = 270000
            

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
                        input_modelweights_folder = f"/<dirname>/<name>/affine_tranformation/objects/DINO/regressor/Linear_Regression/output/llava-v1.5-13b/{category}/{blended_shape}/{angle_str_plural}/{scale}/_visionEncAll/clip_vitl336/what_angle/72_test_samples/Scaled/RidgeRegression/train_and_test"
                        input_file_sin = f"{input_modelweights_folder}/sin_model_weights.csv"
                        input_file_cos = f"{input_modelweights_folder}/cos_model_weights.csv"
                        #Reference embedding file
                        ref_emb = f"/<dirname>/<name>/g-t_embedding_classifier/llava/LLaVA/llava/eval/pca_images/what_angle/{angle_str_singular}/{blended_shape}/{scale}/embeddings/vision_model.encoder.layers.23.layer_norm2.npy"
                        #Out dirs for new embeddings
                        out_emb_dir = f"/<dirname>/<name>/g-t_embedding_classifier/llava/LLaVA/llava/eval/pca_images/what_angle/{angle_str_singular}/{blended_shape}/{scale}_{pair_str}_{num_bins}modelweight/embeddings"
                        # out_origlogs_dir = f"/<dirname>/<name>/affine_tranformation/objects/DINO/regressor/Linear_Regression/output/llava-v1.5-13b/{category}/{blended_shape}/{angle_str_plural}/{scale}/_visionEncAll/clip_vitl336/what_angle/72_test_samples/Scaled/RidgeRegression/train_and_test"
                    elif(modelname == "llava-v1.6-vicuna-13b"):
                        #Model weight files
                        input_modelweights_folder = f"/<dirname>/<name>/affine_tranformation/objects/DINO/regressor/Linear_Regression/output/llava-v1.6-vicuna-13b/{category}/{blended_shape}/{angle_str_plural}/{scale}/_visionEncAll/clip_vitl336/what_angle/72_test_samples/Scaled/RidgeRegression/train_and_test"
                        input_file_sin = f"{input_modelweights_folder}/sin_model_weights.csv"
                        input_file_cos = f"{input_modelweights_folder}/cos_model_weights.csv"
                        #Reference embedding file
                        ref_emb = f"/<dirname>/<name>/g-t_embedding_classifier/llava1.6/LLaVA/llava/eval/pca_images/what_angle/{angle_str_singular}/{blended_shape}/{scale}/embeddings/vision_model.encoder.layers.23.layer_norm2.npy"
                        #Out dirs for new embeddings
                        out_emb_dir = f"/<dirname>/<name>/g-t_embedding_classifier/llava1.6/LLaVA/llava/eval/pca_images/what_angle/{angle_str_singular}/{blended_shape}/{scale}_{pair_str}_{num_bins}modelweight/embeddings"
                        # out_logs_dir = "/<dirname>/<name>/affine_tranformation/objects/DINO/regressor/Linear_Regression/output/LLAVA/llava-v1.6-vicuna-13b/logs"
                    os.makedirs(out_emb_dir, exist_ok=True)

                    pd.set_option('display.float_format', lambda x: '%.20f' % x)

                    # # Read files, skip first 2 rows, take only the second column
                    # df_sin = pd.read_csv(file_sin, skiprows=2, header=None, usecols=[1])
                    # df_cos = pd.read_csv(file_cos, skiprows=2, header=None, usecols=[1])

                    # # Rename column for clarity
                    # df_sin.columns = ["Weights"]
                    # df_cos.columns = ["Weights"]

                    # Read files, skip first 2 rows, take both columns (index + weight)
                    df_sin = pd.read_csv(input_file_sin, skiprows=2, header=None, usecols=[0, 1])
                    df_cos = pd.read_csv(input_file_cos, skiprows=2, header=None, usecols=[0, 1])

                    # Rename columns for clarity
                    df_sin.columns = ["Index", "Weights"]
                    df_cos.columns = ["Index", "Weights"]

                    # Compute top 10 by magnitude
                    # top10_sin = df_sin.reindex(df_sin["Weights"].abs().nlargest(10).index)
                    # top10_cos = df_cos.reindex(df_cos["Weights"].abs().nlargest(10).index)

                    # Top rows by absolute weight, keeping index and weights
                    # top_sin = df_sin.loc[df_sin["Weights"].abs().nlargest(num_bins).index]
                    # top_cos = df_cos.loc[df_cos["Weights"].abs().nlargest(num_bins).index]
                    # Compute average of absolute weights
                    df_avg_abs = df_sin.copy()
                    df_avg_abs["AvgAbs_Weight"] = (df_sin["Weights"].abs() + df_cos["Weights"].abs()) / 2

                    # Select top N based on this average absolute weight
                    top_avg_abs = df_avg_abs.loc[df_avg_abs["AvgAbs_Weight"].nlargest(num_bins).index]
                    # print(top_avg_abs)
                    # sys.exit(1)

                    
                    # #PLOT HISTOGRAMS IF NOT ALREADY SAVED
                    # hist_path = f"{input_modelweights_folder}/model_weights_histogram.png"
                    # print("Hist path : ", hist_path)
                    # print("Data min:", np.min(df_sin["Weights"]))
                    # print("Data max:", np.max(df_sin["Weights"]))
                    # # sys.exit(1)
                    # if not os.path.exists(hist_path):
                    #     # Plot histograms
                    #     plt.figure(figsize=(12, 5))
                    #     # num_hist_bins = 100
                    #     plt.subplot(1, 2, 1)
                    #     counts, bin_edges, _ = plt.hist(df_sin["Weights"], bins="auto", edgecolor='black')
                        
                    #     # sys.exit(1)
                    #     # counts, bin_edges, _ = plt.hist(log_weights_sin, bins=num_bins, edgecolor='black')
                    #     for j in range(len(counts)):
                    #         # print(" j is : ", j)
                    #         bin_start = bin_edges[j]
                    #         bin_end = bin_edges[j + 1]
                    #         count = counts[j]
                    #         # print(f"  Bin {j + 1}: Range = [{bin_start:.4f}, {bin_end:.4f}), Count = {int(count)}")
                    #         # Choose format depending on bin size
                    #         if abs(bin_end - bin_start) < 1e-3:
                    #             fmt = "{:.8e}"   # scientific
                    #         else:
                    #             fmt = "{:.4f}"   # fixed-point

                    #         print(f"  Bin {j+1}: Range = [{fmt.format(bin_start)}, {fmt.format(bin_end)}), Count = {int(count)}")

                    #     print("----------------------------------------------------------")
                    #     plt.title("Histogram - Sin Model Weights")
                    #     plt.xlabel("Weight Values")
                    #     plt.ylabel("Count")
                    #     # Force scientific notation on x-axis if needed
                    #     plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
                    #     # plt.xscale("log")

                    #     plt.subplot(1, 2, 2)
                    #     counts, bin_edges, _ = plt.hist(df_cos["Weights"], bins="auto", edgecolor='black')
                    #     # counts, bin_edges, _ = plt.hist(log_weights_cos, bins=num_bins, edgecolor='black')
                    #     plt.title("Histogram - Cos Model Weights")
                    #     plt.xlabel("Weight Values")
                    #     plt.ylabel("Count")
                    #     plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
                    #     # plt.xscale("log")
                    #     for j in range(len(counts)):
                    #         # print(" j is : ", j)
                    #         bin_start = bin_edges[j]
                    #         bin_end = bin_edges[j + 1]
                    #         count = counts[j]
                    #         if abs(bin_end - bin_start) < 1e-3:
                    #             fmt = "{:.8e}"   # scientific
                    #         else:
                    #             fmt = "{:.4f}"   # fixed-point
                    #         # print(f"  Bin {j + 1}: Range = [{bin_start:.4f}, {bin_end:.4f}), Count = {int(count)}")
                    #         print(f"  Bin {j+1}: Range = [{fmt.format(bin_start)}, {fmt.format(bin_end)}), Count = {int(count)}")

                    #     plt.tight_layout()
                    #     plt.show()
                    #     # plt.savefig(f"{out_logs_dir}/modelweight/{blended_shape}/{angle_str_singular}/{scale}/model_weights_histogram.png")
                    #     plt.savefig(f"{input_modelweights_folder}/model_weights_histogram.png")


                    #----------------------------------------------------------------
                    #COPY VALUES FROM LOCATIONS IN EMBEDDING 3 TO EMBEDDING 15
                    #----------------------------------------------------------------

                    # ref_emb = f"/<dirname>/<name>/g-t_embedding_classifier/llava/LLaVA/llava/eval/pca_images/what_angle/1_degree/dog_on_beach/{scale}/embeddings/vision_model.encoder.layers.23.layer_norm2.npy"
                    # Load embedding
                    embedding = np.load(ref_emb, allow_pickle=True).item()
                    
                    embedding_copy = {k: v.clone() for k, v in embedding.items()}
                
                    # Extract indices (assuming they are the DataFrame index)
                    # sin_indices = top_sin.index.to_list()
                    # cos_indices = top_cos.index.to_list()

                    # common_indices = set(sin_indices) & set(cos_indices)
                    # num_common = len(common_indices)

                    # print("Number of unique indices in sin model:", len(sin_indices))
                    # print("Number of unique indices in cos model:", len(cos_indices))
                    # print("Number of common indices:", num_common)

                    # Combine unique indices
                    # target_indices = list(set(sin_indices + cos_indices))
                    target_indices = top_avg_abs["Index"].to_list()
                    # print(target_indices)
                    # sys.exit(1)
                    

                    # Step 3: Replace values in row 14 with values from row 2 at those indices
                    if(modelname == "llava-v1.5-13b"):
                        for idx in target_indices:
                            row = idx // 1024
                            col = idx % 1024
                            embedding_copy[str(num)][0, row, col] = embedding[str(anchor)][0, row, col]
                    elif(modelname == "llava-v1.6-vicuna-13b"):
                    
                        num_rows, num_cols = embedding[str(anchor)].shape[1], embedding[str(anchor)].shape[2]  # (576, 1024)
                        for idx in target_indices:
                            # decode flattened index (batch, row, col)
                            batch = idx // (num_rows * num_cols)
                            rem = idx % (num_rows * num_cols)
                            row = rem // num_cols
                            col = rem % num_cols

                            embedding_copy[str(num)][batch, row, col] = embedding[str(anchor)][batch, row, col]

                    # Step 4: Save as new .npy file
                    # new_emb_path = ref_emb.replace(".npy", "_modified.npy")
                    new_emb_name = "vision_model.encoder.layers.23.layer_norm2.npy"
                    final_emb_path = os.path.join(out_emb_dir, new_emb_name)
                    print("Final emb path : ", final_emb_path)
                    np.save(final_emb_path, embedding_copy)

                    print(f"Modified embedding saved at {final_emb_path}")
                    print(f"Completed processing for num_bins : {num_bins}")