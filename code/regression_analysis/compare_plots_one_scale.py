import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plot_all_values = False

shape = "blended"
modelnames = ["llava-v1.5-13b", "llava-v1.6-vicuna-13b"]
scale_str = 'Scaled'
subset = "_visionEncAll"
qs_type = "what_angle"
backbone = "clip_vitl336"
angle = '1_degrees_FIXED'
model_type = 'RidgeRegression'
angle_repr = 'Degrees'
plot_subfolder_name = 'gtVSerror'

foreground_list = ['dog', 'lizard', 'train']
background_list = ['beach', 'fish', 'indoor']

# Only one scale
scales_to_plot = [3] #0,3,15
scales_legend = [2] #1,2,3
required_shape = "dog_on_beach" #"dog_on_beach", "lizard_on_fish", "train_on_indoor"

csv_files = ['vision_model.encoder.layers.23.layer_norm2']
legend_names = ['encoder.layers.23.layer_norm2']
num_samples = 360
test_size = str(int(.2*num_samples)) + '_test_samples'

plot_gtVSerror_dir = (
    "/<dirname>/<name>/affine_tranformation/objects/"
    "DINO/regressor/Linear_Regression/regression_analysis/regression_comparison_plots"
)
os.makedirs(plot_gtVSerror_dir, exist_ok=True)

markers = ['o', 's']
colors = ['b', 'g']

if plot_all_values:

    for subshape in [f"{f}_on_{b}" for f in foreground_list for b in background_list]:
        print(f"Processing subshape: {subshape}")
        if required_shape not in subshape:
            continue

        # --- Single Figure ---
        fig, ax = plt.subplots(figsize=(12, 6))

        for scale, scale_label in zip(scales_to_plot, scales_legend):
            scaleBlendedDir = f"scale{scale + 1}"

            for midx, model in enumerate(modelnames):
                parent_dir = (
                    f"/<dirname>/<name>/affine_tranformation/objects/"
                    f"DINO/regressor/Linear_Regression/output/{model}/{shape}/{subshape}/"
                    f"{angle}/{scaleBlendedDir}/{subset}/{backbone}/{qs_type}/{test_size}/{scale_str}/"
                    f"{model_type}/train_and_test"
                )
                file = f"{csv_files[0]}_{angle_repr}_labels_and_predictions.csv"
                file_path = os.path.join(parent_dir, file)

                if not os.path.exists(file_path):
                    print(f"Missing file for {model}: {file_path}")
                    continue

                df = pd.read_csv(file_path)

                ax.plot(
                    df.iloc[:-3, 0],
                    df.iloc[:-3, 2],
                    label=f"{model}",
                    marker=markers[midx],
                    linestyle='-',
                    color=colors[midx],
                    markersize=3,
                    linewidth=1.8
                )

        # --- Formatting ---
        ax.set_title(f"{subshape} (Scale {scales_legend[0]})", fontsize=11)
        ax.set_xlabel('Ground Truth Angle', fontsize=10)
        ax.set_ylabel('Absolute Angle Difference (Actual - Predicted)', fontsize=10)
        ax.grid(True)
        ax.tick_params(axis='x', labelsize=8, rotation=90)
        ax.tick_params(axis='y', labelsize=9)
        ax.legend(fontsize=8, loc='upper right')

        # --- Save Figure ---
        scale_str_joined = "_".join(str(s) for s in scales_to_plot)
        savepath = os.path.join(
            plot_gtVSerror_dir,
            f"{subshape}_scale{scale_str_joined}_compare_regression_1.5_vs_1.6.png"
        )
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
else:
    for subshape in [f"{f}_on_{b}" for f in foreground_list for b in background_list]:
        if required_shape not in subshape:
            continue

        # --- Smaller Figure for single-column paper ---
        fig, ax = plt.subplots(figsize=(6, 4))  # compact figure for paper column

        for scale, scale_label in zip(scales_to_plot, scales_legend):
            scaleBlendedDir = f"scale{scale + 1}"

            for midx, model in enumerate(modelnames):
                parent_dir = (
                    f"/<dirname>/<name>/affine_tranformation/objects/"
                    f"DINO/regressor/Linear_Regression/output/{model}/{shape}/{subshape}/"
                    f"{angle}/{scaleBlendedDir}/{subset}/{backbone}/{qs_type}/{test_size}/{scale_str}/"
                    f"{model_type}/train_and_test"
                )
                file = f"{csv_files[0]}_{angle_repr}_labels_and_predictions.csv"
                file_path = os.path.join(parent_dir, file)

                if not os.path.exists(file_path):
                    print(f"Missing file for {model}: {file_path}")
                    continue

                df = pd.read_csv(file_path)
                df_all = df.iloc[:-3]

                # --- Plot actual 72 data points ---
                ax.plot(
                    df_all.iloc[:, 0],
                    df_all.iloc[:, 2],
                    label=f"{model}",
                    marker=markers[midx],
                    linestyle='-',
                    color=colors[midx],
                    markersize=4,
                    linewidth=1.6
                )

        # --- Formatting for single-column readability ---
        ax.set_title(f"{subshape} (Scale {scales_legend[0]})", fontsize=16, fontweight='bold')
        ax.set_xlabel('Ground Truth Angle', fontsize=14)
        ax.set_ylabel('Abs(Actual - Predicted)', fontsize=14)
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5)

        # --- Evenly spaced labels (0, 30, ..., 330, 359) but NOT tied to data ---
        xticklabels = list(range(0, 331, 30)) + [359]
        xticks = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], len(xticklabels))

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.tick_params(axis='x', labelsize=12, rotation=45)
        ax.tick_params(axis='y', labelsize=12)

        # --- Add vertical gray guide lines at those label positions ---
        for x in xticks:
            ax.axvline(x=x, color='gray', linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)

        ax.legend(fontsize=12, loc='lower left', frameon=True)

        # --- Save Figure ---
        scale_str_joined = "_".join(str(s) for s in scales_to_plot)
        savepath = os.path.join(
            plot_gtVSerror_dir,
            f"{subshape}_scale{scale_str_joined}_compare_regression_1.5_vs_1.6_fixed.png"
        )
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        plt.close(fig)

