import os
import sys
import math
import pandas as pd
sys.path.append("/s/red/a/nobackup/vision/anju/affine_tranformation/objects/DINO/regressor/Linear_Regression")
from utils import *
import matplotlib.pyplot as plt
from scipy.stats import chi2

def main(category, subset, shape_, scaleBlendedDir, modelname, subdir1=None):

    # if(shape_==None):
    #     shape = "bird"  # rectangle, hexagon, star, triangle, insect, bird, dog, rectangle-triangle, rectangle-triangle_visionEncAll, "blended"
    # else:
    #     shape = shape_
    #     category = "blended"
    # shape = 'bird' #ampersand, malayalamA, tree, rectangle-triangle, insect, bird, dog, rectangle
    shape = shape_
    scale_str = 'Scaled' # Unscaled, Scaled
    # model = "LLAVA"
    model=modelname
    # subset = "_visionEncAll" #_llavaAll, _visionEncAll, ComplVisionEnc
    qs_type = "what_angle" #["what_angle", "what_is_color"]
    perform_chi_squared_test = False
    perform_shapiro_wilk_test = False
    perform_kolmogorov_smirnov_test = True
    perform_cramervonmises_test = False

    if(modelname == "llava-v1.5-13b" or modelname == "llava-v1.6-vicuna-13b"):
        backbone = "clip_vitl336" #Resnet50 (DINO), clip_vitl336 (llava)
    elif(modelname == "llava-ov-qwen2-7b"):
        backbone = "siglip"
    elif(modelname == "Qwen2.5-VL-7B-Instruct"):
        backbone = "vit"
    angle = '1_degrees_FIXED' # 05_degrees, 1_degrees, 2_degrees, 5_degrees, 10_degrees
    model_type = 'RidgeRegression' #RidgeRegression, MLP
    angle_repr = 'Degrees' #Degrees/Radians'
    plot_subfolder_name = "stat_test"

    # test_samples = '36_test_samples'

    if angle =='05_degrees':
        num_samples = 720
    elif angle =='1_degrees':
        if(shape=='rectangle'):
            num_samples = 180
        elif(shape=='triangle'):
            num_samples = 120
        else:
            num_samples = 360
    elif angle =='1_degrees_FIXED':
        if(category=="blended"):
            num_samples = 360
        elif(category=="natural_cropped" or category=="in_place_rotated"):
            num_samples = 180
    elif angle =='2_degrees':
        num_samples = 180
    elif angle =='5_degrees':
        num_samples = 72
    elif angle =='10_degrees':
        num_samples = 36

    if(category=="blended"):
        if(subset == "ComplVisionEnc"):
            csv_files = ['model.embed_tokens',
            'model.layers.0.post_attention_layernorm',
            'model.layers.39.post_attention_layernorm',
            'model.norm',
            'model.mm_projector.2',
            'lm_head']

        elif(subset == "_visionEncAll"):
            # csv_files = ['vision_model.embeddings.patch_embedding', 'vision_model.embeddings.position_embedding', 'vision_model.pre_layrnorm',
            #             'vision_model.encoder.layers.0.self_attn.k_proj', 'vision_model.encoder.layers.0.self_attn.v_proj', 'vision_model.encoder.layers.0.self_attn.q_proj', 
            #             'vision_model.encoder.layers.0.self_attn.out_proj', 'vision_model.encoder.layers.0.layer_norm1', 'vision_model.encoder.layers.0.mlp.activation_fn',
            #             'vision_model.encoder.layers.0.mlp.fc1', 'vision_model.encoder.layers.0.mlp.fc2', 'vision_model.encoder.layers.0.layer_norm2',
            #             'vision_model.encoder.layers.23.self_attn.k_proj', 'vision_model.encoder.layers.23.self_attn.v_proj', 'vision_model.encoder.layers.23.self_attn.q_proj',
            #             'vision_model.encoder.layers.23.self_attn.out_proj', 'vision_model.encoder.layers.23.layer_norm1', 'vision_model.encoder.layers.23.mlp.activation_fn',
            #             'vision_model.encoder.layers.23.mlp.fc1', 'vision_model.encoder.layers.23.mlp.fc2', 'vision_model.encoder.layers.23.layer_norm2',
            #             'vision_model.post_layernorm']

            # csv_files = ['vision_model.encoder.layers.23.layer_norm2','vision_model.post_layernorm']
            # csv_files = ['vision_model.post_layernorm']
            csv_files = ['vision_model.encoder.layers.23.layer_norm2']
    elif(category=="natural_cropped" or category=="in_place_rotated"):
        csv_files = ["vision_tower"]
    

    test_size = str(int(.2*num_samples)) + '_test_samples'
    # parent_dir = f"/s/red/a/nobackup/vision/anju/affine_tranformation/objects/DINO/regressor/Linear_Regression/output/{model}/{shape}/{subset}/{backbone}/{qs_type}/{angle}/{test_size}/{scale_str}/{model_type}/train_and_test"
    # output_dir = f"/s/red/a/nobackup/vision/anju/affine_tranformation/objects/DINO/regressor/Linear_Regression/output/{model}/{shape}/{subset}/{backbone}/{qs_type}/{angle}/{test_size}/{scale_str}/{model_type}/analysis"
    # plot_dir = f"/s/red/a/nobackup/vision/anju/affine_tranformation/objects/DINO/regressor/Linear_Regression/output/plots/{model}/{shape}/{subset}/{backbone}/{qs_type}/{angle}/{test_size}/{scale_str}/{model_type}/stat_test"
    if(category=="blended"):
        if(shape_==None):
            parent_dir = f"/s/red/a/nobackup/vision/anju/affine_tranformation/objects/DINO/regressor/Linear_Regression/output/{model}/{shape}/{subset}/{backbone}/{qs_type}/{angle}/{test_size}/{scale_str}/{model_type}/train_and_test"
            output_dir = f"/s/red/a/nobackup/vision/anju/affine_tranformation/objects/DINO/regressor/Linear_Regression/output/{model}/{shape}/{subset}/{backbone}/{qs_type}/{angle}/{test_size}/{scale_str}/{model_type}/analysis"
            plot_dir = f"/s/red/a/nobackup/vision/anju/affine_tranformation/objects/DINO/regressor/Linear_Regression/output/plots/{model}/{shape}/{subset}/{backbone}/{qs_type}/{angle}/{test_size}/{scale_str}/{model_type}/{plot_subfolder_name}"
        else:
            parent_dir = f"/s/red/a/nobackup/vision/anju/affine_tranformation/objects/DINO/regressor/Linear_Regression/output/{model}/{category}/{shape}/{angle}/{scaleBlendedDir}/{subset}/{backbone}/{qs_type}/{test_size}/{scale_str}/{model_type}/train_and_test"
            output_dir = f"/s/red/a/nobackup/vision/anju/affine_tranformation/objects/DINO/regressor/Linear_Regression/output/{model}/{category}/{shape}/{angle}/{scaleBlendedDir}/{subset}/{backbone}/{qs_type}/{test_size}/{scale_str}/{model_type}/analysis"
            plot_dir = f"/s/red/a/nobackup/vision/anju/affine_tranformation/objects/DINO/regressor/Linear_Regression/output/plots/{model}/{category}/{shape}/{angle}/{scaleBlendedDir}/{subset}/{backbone}/{qs_type}/{test_size}/{scale_str}/{model_type}/{plot_subfolder_name}"
    elif(category=="natural_cropped" or category=="in_place_rotated"):
            parent_dir = f"/s/red/a/nobackup/vision/anju/affine_tranformation/objects/DINO/regressor/Linear_Regression/output/{model}/{category}/{shape}/{angle}/{scaleBlendedDir}/{subset}/{backbone}/{qs_type}/{test_size}/{scale_str}/{model_type}/{subdir1}"
            output_dir = f"/s/red/a/nobackup/vision/anju/affine_tranformation/objects/DINO/regressor/Linear_Regression/output/{model}/{category}/{shape}/{angle}/{scaleBlendedDir}/{subset}/{backbone}/{qs_type}/{test_size}/{scale_str}/{model_type}/analysis"
            plot_dir = f"/s/red/a/nobackup/vision/anju/affine_tranformation/objects/DINO/regressor/Linear_Regression/output/plots/{model}/{category}/{shape}/{angle}/{scaleBlendedDir}/{subset}/{backbone}/{qs_type}/{test_size}/{scale_str}/{model_type}/{plot_subfolder_name}"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    filename = "analysis.txt"
    f_output = open(os.path.join(output_dir, filename), "w")
    sum_abs_errors = []
    for layer_index, entry in enumerate(csv_files):
        print("**************************************************************************************")
        print("File Name : ", entry, "shape : ",shape)
        print("**************************************************************************************\n")
        f_output.write("File Name : " + entry + "\n")
        file = entry + f"_{angle_repr}_labels_and_predictions.csv"
        file_path = os.path.join(parent_dir, file)
        df = pd.read_csv(file_path, delimiter=",")
        sum_abs_error = df.iloc[-2, 0]
        if("Sum Absolute Error on Test Set:" in str(sum_abs_error)):
            sum_abs_error = float(sum_abs_error.split(":")[-1].split()[0].strip())
            print("-> Sum Absolute Error on Test Set : ", sum_abs_error)
            f_output.write("Sum Absolute Error on Test Set : " + str(sum_abs_error) + "\n")
            sum_abs_errors.append(sum_abs_error)
        r2score = df.iloc[-1, 0]
        if("R^2 Score on Test Set:" in str(r2score)):
            r2score = float(r2score.split(":")[-1].strip())
            print("-> R^2 Score on Test Set : ", r2score)
            f_output.write("R^2 Score on Test Set : " + str(r2score) + "\n")
        # print("-----------------------------------------------------\n")
        f_output.write("-----------------------------------------------------\n")

        pred_vals = df.iloc[:-3, 2]
        plot_layer_dir = f"{plot_dir}/{layer_index+1}_{entry}"
        os.makedirs(plot_layer_dir, exist_ok=True)
        

        # *********** BEGIN CHI-SQUARED TEST ***********
        if(perform_chi_squared_test):
            log_file_name = f"chi_squared_test_{entry}.txt"
            # file_handle = open(os.path.join(plot_layer_dir, log_file_name), "w")
            file_path = os.path.join(plot_layer_dir, log_file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
            file_handle = open(file_path, "a")

            groups, edges, cdf_by_group = group_by_dynamic_range(pred_vals)

            total_count = sum(len(values) for values in groups.values())
            chi_squared_terms = []

            for group_label, values in groups.items():
                observed = len(values)
                expected = total_count * cdf_by_group[group_label]

                # Avoid division by zero
                if expected > 0:
                    chi_term = ((observed - expected) ** 2) / expected
                    chi_squared_terms.append(chi_term)

                    # print(f"{group_label}:")
                    # print(f"  Values: {values}")
                    # print(f"  Count: {observed}")
                    # print(f"  CDF: {cdf_by_group[group_label]:.4f}")
                    # print(f"  Expected: {expected:.2f}")
                    # print(f"  (obs - exp)^2 / exp: {chi_term:.4f}")
                    # print()

            # Total chi-squared-like statistic
            chi_squared_sum = sum(chi_squared_terms)
            print(f"Total sum of (count - expected)^2 / expected = {chi_squared_sum:.4f}")
            file_handle.write(f"Total sum of (count - expected)^2 / expected = {chi_squared_sum:.4f}\n")
        
            df = len(groups) - 1 - 2 # degrees of freedom
            p_value = 1 - chi2.cdf(chi_squared_sum, df)

            for group_label, values in groups.items():
                print(f"{group_label}:")
                file_handle.write(f"{group_label}:\n")
                print(f"  Values: {values}")
                file_handle.write(f"  Values: {values}\n")
                print(f"  Count: {len(values)}")
                file_handle.write(f"  Count: {len(values)}\n")

            print(f"Number of bins: {len(groups)}")
            file_handle.write(f"Number of bins: {len(groups)}\n")
            print(f"Degrees of Freedom: {df}")
            file_handle.write(f"Degrees of Freedom: {df}\n")
            print(f"Chi-squared Sum: {chi_squared_sum:.4f}")
            file_handle.write(f"Chi-squared Sum: {chi_squared_sum:.4f}\n")
            print(f"P-value: {p_value:.6f}")
            file_handle.write(f"P-value: {p_value:.6f}\n")

            alpha = 0.05  # significance level
            print(f"Significance Level: {alpha}")
            file_handle.write(f"Significance Level: {alpha}\n")
            if p_value < alpha:
                print(f"\nP-value ({p_value:.4f}) is less than the significance level ({alpha}).")
                file_handle.write(f"\nP-value ({p_value:.4f}) is less than the significance level ({alpha}).\n")
                print("Reject the null hypothesis: The data does not follow the normal distribution.")
                file_handle.write("Reject the null hypothesis: The data does not follow the normal distribution.\n")
            else:
                print(f"\nP-value ({p_value:.4f}) is greater than or equal to the significance level ({alpha}).")
                file_handle.write(f"\nP-value ({p_value:.4f}) is greater than or equal to the significance level ({alpha}).\n")
                print("Fail to reject the null hypothesis: There is not enough evidence to suggest the data does not follow the normal distribution.")
                file_handle.write("Fail to reject the null hypothesis: There is not enough evidence to suggest the data does not follow the normal distribution.\n")

        # *********** END CHI-SQUARED TEST ***********

        # *********** BEGIN SHAPIRO-WILK NORMALITY TEST ***********
        if(perform_shapiro_wilk_test):
            log_file_name = f"shapiro_wilk_test_{entry}.txt"
            
            file_handle_shapiro = open(os.path.join(plot_layer_dir, log_file_name), "w")
            file_handle_shapiro = open(os.path.join(plot_layer_dir, log_file_name), "a")
            perform_shapiro_wilk_normality_test(pred_vals, file_handle_shapiro)
        # ******** END SHAPIRO-WILK NORMALITY TEST **************

        # *********** BEGIN Kolmogorov Smirnov NORMALITY TEST ***********
        if(perform_kolmogorov_smirnov_test):
            log_file_name = f"kolmogorov-smirnov_test_{entry}.txt"
            
            file_handle_ks = open(os.path.join(plot_layer_dir, log_file_name), "w")
            file_handle_ks = open(os.path.join(plot_layer_dir, log_file_name), "a")
            perform_kolmogorov_smirnov_normality_test(pred_vals, file_handle_ks)
        # ******** END Kolmogorov Smirnov NORMALITY TEST **************

        # *********** BEGIN CRAMER-VON-MISES TEST ***********
        if(perform_cramervonmises_test):
            log_file_name = f"vonMises_test_{entry}.txt"
            file_handle_vonMises = open(os.path.join(plot_layer_dir, log_file_name), "w")
            file_handle_vonMises = open(os.path.join(plot_layer_dir, log_file_name), "a")
            perform_cramervonmises_normality_test(pred_vals, file_handle_vonMises)

        # ********** END CRAMER-VON-MISES TEST **************
        print("-----------------------------------------------------\n")

    #Plotting Sum Absolute Errors
    plt.figure(figsize=(10, 12))  # Adjust height for long label list
    processed_files = ['.'.join(file.split('.')[1:]) for file in csv_files]
    bars = plt.bar(processed_files, sum_abs_errors, color='skyblue')
    plt.xlabel("Layer Name")
    plt.xticks(rotation=90, ha='center')
    plt.ylabel("Sum Absolute Error")
    plt.title("Sum Absolute Error per Layer")

    # Add the value on top of each bar
    for bar in bars:
        # Get the height (value) of each bar
        yval = bar.get_height()
        # Place the text at the top of each bar
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom', rotation=45)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "sum_abs_error_plot.png")

