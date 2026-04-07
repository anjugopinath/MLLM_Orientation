import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import seaborn as sns
import glob
import os

def main(category, shape_, scaleBlendedDir, modelname, subset, required_shape=None, subdir1=None):

    # if(shape_==None):
    #     shape = "bird"  # rectangle, hexagon, star, triangle, insect, bird, dog, rectangle-triangle, rectangle-triangle_visionEncAll, "blended"
    # else:
    #     shape = shape_
    #     category = "blended"

    scale_str = 'Scaled' # Unscaled, Scaled
    # model = "LLAVA"
    model = modelname
    # subset = "_visionEncAll" #_llavaAll, _visionEncAll, ComplVisionEnc
    numLayers = 'LastOne' #All, LastTwo, LastOne
    qs_type = "what_angle" #["what_angle", "what_is_color"]

    if(category == "natural_cropped" or category == "in_place_rotated"):
        if(subset == "visionEncAll"):
            files = ['vision_tower']
    elif(category=="blended"):
        if(subset == "_llavaAll"):
            files = ''
        elif(subset == "_visionEncAll"):
            # files = ['vision_model.post_layernorm']
            files = ['vision_model.encoder.layers.23.layer_norm2']
        elif(subset == "ComplVisionEnc"):
            files = ['model.embed_tokens',
                'model.layers.0.post_attention_layernorm',
                'model.layers.39.post_attention_layernorm',
                'model.norm',
                'model.mm_projector.2',
                'lm_head']

    if(modelname == "llava-v1.5-13b" or modelname == "llava-v1.6-vicuna-13b"):
        backbone = "clip_vitl336" #Resnet50 (DINO), clip_vitl336 (llava)
    elif(modelname == "llava-ov-qwen2-7b"):
        backbone = "siglip"
    elif(modelname == "Qwen2.5-VL-7B-Instruct"):
        backbone = "vit"
    angle = '1_degrees_FIXED' # 05_degrees, 1_degrees, 2_degrees, 5_degrees, 10_degrees
    model_type = 'RidgeRegression' #RidgeRegression, MLP
    angle_repr = 'Degrees' #Degrees/Radians'
    # plot_subfolder_name = 'stat_test'
    # plot_subfolder_name = 'stat_plots'
    if(category=="blended"):
        plot_gtVSerror_dir = 'stat_plots'
    elif(category=="natural_cropped" or category=="in_place_rotated"):
        plot_gtVSerror_dir = os.path.join("stat_plots",f"{modelname}",f"{category}")
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
        if(category == "blended"):
            num_samples = 360
        elif(category == "natural_cropped" or category == "in_place_rotated"):
            num_samples = 180
    elif angle =='2_degrees':
        num_samples = 180
    elif angle =='5_degrees':
        num_samples = 72
    elif angle =='10_degrees':
        num_samples = 36

    test_size = str(int(.2*num_samples)) + '_test_samples'
    # parent_dir = f"/s/red/a/nobackup/vision/anju/affine_tranformation/objects/DINO/regressor/Linear_Regression/output/{model}/{shape}/{subset}/{backbone}/{qs_type}/{angle}/{test_size}/{scale_str}/{model_type}/train_and_test"
    if(category=="blended"):
        if(shape_==None):
            parent_dir = f"/s/red/a/nobackup/vision/anju/affine_tranformation/objects/DINO/regressor/Linear_Regression/output/{model}/{shape}/{subset}/{backbone}/{qs_type}/{angle}/{test_size}/{scale_str}/{model_type}/train_and_test"
        else:
            parent_dir = f"/s/red/a/nobackup/vision/anju/affine_tranformation/objects/DINO/regressor/Linear_Regression/output/{model}/{category}/{shape}/{angle}/{scaleBlendedDir}/{subset}/{backbone}/{qs_type}/{test_size}/{scale_str}/{model_type}/train_and_test"
    elif(category=="natural_cropped" or category=="in_place_rotated"):
        parent_dir = f"/s/red/a/nobackup/vision/anju/affine_tranformation/objects/DINO/regressor/Linear_Regression/output/{model}/{category}/{shape_}/{angle}/scale1/{subset}/{backbone}/{qs_type}/{test_size}/{scale_str}/{model_type}/{subdir1}"
    
    os.makedirs(plot_gtVSerror_dir, exist_ok=True)

    # if(model=="LLAVA"):
    #     if(subset == "_visionEncAll"):
    #         csv_files = ['vision_model.post_layernorm']


    for file in files:
        layername = file
        file = file + f"_{angle_repr}_labels_and_predictions.csv"
        file_path = os.path.join(parent_dir, file)
        print("file_path : ",file_path)
        
        df = pd.read_csv(file_path, delimiter=",")  # Read CSV file
        data = df.iloc[:-3, 2]

        print("data : ",data)
        print("data shape : ",data.shape)

        # Calculate skewness and kurtosis
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)  # Fisher's definition (normal => 0)
        mean = np.mean(data)
        std_dev = np.std(data, ddof=1)  # sample standard deviation
        variance = np.var(data, ddof=1)  # sample variance

        cv = (std_dev / mean) * 100

        # n = len(data)
        # # Standard Errors
        # SE_skew = np.sqrt(6 / n)
        # SE_kurt = np.sqrt(24 / n)

        # # Z-scores
        # z_skew = skewness / SE_skew
        # z_kurt = kurtosis / SE_kurt

        # # Two-tailed p-values
        # p_skew = 2 * (1 - stats.norm.cdf(abs(z_skew)))
        # p_kurt = 2 * (1 - stats.norm.cdf(abs(z_kurt)))

        print("layername : ",layername)
        print(f"Mean: {mean:.4f}")
        print(f"Standard Deviation: {std_dev:.4f}")
        print(f"CV = {cv:.2f}%")
        print(f"Variance: {variance:.4f}")
        print(f"Skewness: {skewness:.4f}")
        print(f"Kurtosis: {kurtosis:.4f}")
        # print(f"Z-skewness: {z_skew:.4f}, p-value: {p_skew:.4f}")
        # print(f"Z-kurtosis: {z_kurt:.4f}, p-value: {p_kurt:.4f}")

        # Using skewtest to get the Z-statistic
        z_skew, p_skew = stats.skewtest(data)
        print(f"Z-score for skewness (from skewtest): {z_skew:.4f}")
        print(f"P-value for skewness test: {p_skew:.4f}")

        # Using kurtosistest to get the Z-statistic
        z_kurt, p_kurt = stats.kurtosistest(data)
        print(f"\nZ-score for kurtosis (from kurtosistest): {z_kurt:.4f}")
        print(f"P-value for kurtosis test: {p_kurt:.4f}")

        # # Set up the figure
        # plt.figure(figsize=(14, 10))

        # # Q-Q Plot
        # plt.subplot(2, 2, 1)
        # stats.probplot(data, dist="norm", plot=plt)
        # plt.title("Q-Q Plot")

        # # Histogram with KDE
        # plt.subplot(2, 2, 2)
        # sns.histplot(data, kde=True, bins=15, color='skyblue')
        # plt.title("Histogram with KDE")
        # plt.xlabel("Data Values")
        # plt.ylabel("Frequency")

        # # P-P Plot
        # plt.subplot(2, 2, 3)
        # sorted_data = np.sort(data)
        # ecdf = np.arange(1, len(data) + 1) / len(data)
        # theoretical_cdf = stats.norm.cdf(sorted_data, loc=mean, scale=std_dev)
        # plt.plot(theoretical_cdf, ecdf, marker='o', linestyle='', label='Empirical vs Theoretical')
        # plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='1:1 Line')
        # plt.title("P-P Plot")
        # plt.xlabel("Theoretical CDF")
        # plt.ylabel("Empirical CDF")
        # plt.legend()

        # # Box Plot
        # plt.subplot(2, 2, 4)
        # sns.boxplot(x=data, color='lightgreen')
        # plt.title("Box Plot")
        # plt.xlabel("Data Values")

        # # Save and close
        # print("plot_gtVSerror_dir",plot_gtVSerror_dir)
        # output_plot_path = os.path.join(plot_gtVSerror_dir, f"qq_pp_hist_box_{file}.png")
        # plt.tight_layout()
        # plt.savefig(output_plot_path, dpi=300)
        # plt.close()
        # Set up the figure
        plt.figure(figsize=(14, 10))

        # Define font sizes for consistency
        title_font = 22
        label_font = 20
        tick_font = 18
        legend_font = 18

        # Q-Q Plot
        plt.subplot(2, 2, 1)
        stats.probplot(data, dist="norm", plot=plt)
        plt.title("Q-Q Plot", fontsize=title_font, fontweight='bold')
        plt.xlabel("Theoretical Quantiles", fontsize=label_font, fontweight='bold')
        plt.ylabel("Ordered Values", fontsize=label_font, fontweight='bold')
        plt.xticks(fontsize=tick_font)
        plt.yticks(fontsize=tick_font)

        # Histogram with KDE
        plt.subplot(2, 2, 2)
        sns.histplot(data, kde=True, bins=15, color='skyblue')
        plt.title("Histogram with KDE", fontsize=title_font, fontweight='bold')
        plt.xlabel("Data Values", fontsize=label_font, fontweight='bold')
        plt.ylabel("Frequency", fontsize=label_font, fontweight='bold')
        plt.xticks(fontsize=tick_font)
        plt.yticks(fontsize=tick_font)

        # P-P Plot
        plt.subplot(2, 2, 3)
        sorted_data = np.sort(data)
        ecdf = np.arange(1, len(data) + 1) / len(data)
        theoretical_cdf = stats.norm.cdf(sorted_data, loc=mean, scale=std_dev)
        plt.plot(theoretical_cdf, ecdf, marker='o', linestyle='', label='Empirical vs Theoretical', markersize=5)
        plt.legend(fontsize=14, prop={'weight': 'bold'})

        plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='1:1 Line')
        plt.title("P-P Plot", fontsize=title_font, fontweight='bold')
        plt.xlabel("Theoretical CDF", fontsize=label_font, fontweight='bold')
        plt.ylabel("Empirical CDF", fontsize=label_font, fontweight='bold')
        plt.xticks(fontsize=tick_font)
        plt.yticks(fontsize=tick_font)
        plt.legend(fontsize=legend_font, loc='lower right')

        # Box Plot
        plt.subplot(2, 2, 4)
        sns.boxplot(x=data, color='lightgreen')
        plt.title("Box Plot", fontsize=title_font, fontweight='bold')
        plt.xlabel("Data Values", fontsize=label_font, fontweight='bold')
        plt.xticks(fontsize=tick_font)
        plt.yticks(fontsize=tick_font)

        # Save and close
        plt.tight_layout()
        if(category=="blended"):
            output_plot_path = os.path.join(plot_gtVSerror_dir, f"{modelname}_{required_shape}_{scaleBlendedDir}_qq_pp_hist_box_{file}_FIXED.png")
        elif(category=="natural_cropped" or category=="in_place_rotated"):
            output_plot_path = os.path.join(plot_gtVSerror_dir, f"{modelname}_{shape_}_{file}.png")
        plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
