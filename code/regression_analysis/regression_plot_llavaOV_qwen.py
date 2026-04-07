import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plot_all_values = False

category = "in_place_rotated" # "natural_cropped" or "in_place_rotated"
modelnames = ["llava-ov-qwen2-7b","Qwen2.5-VL-7B-Instruct"]
scale_str = 'Scaled'
scaleBlendedDir = 'scale1'
subset = "visionEncAll"
qs_type = "what_angle"

angle = '1_degrees_FIXED'
model_type = 'RidgeRegression'
angle_repr = 'Degrees'
plot_subfolder_name = 'gtVSerror'
subdir1 = 'train_and_test_visionEnc'
num_samples = 180
test_size = str(int(.2*num_samples)) + '_test_samples'

if category == "natural_cropped":
    shape_list = ["dog", "lizard", "train", "fish", "indoor", "beach"]
elif category == "in_place_rotated":
    shape_list = ["koala-beach", "vase-indoor", "vase-toaster-indoor"]
else:
    shape_list = []

markers = ['o', 's']
colors = ['b', 'g']

# -----------------------------------
# MAIN LOOP
# -----------------------------------
for subshape in shape_list:

    fig, ax = plt.subplots(figsize=(6, 4))
    x_values_for_ticks = None  # safe initialization

    for midx, model in enumerate(modelnames):

        if(model=="llava-ov-qwen2-7b"):
            backbone = "siglip"
        elif(model=="Qwen2.5-VL-7B-Instruct"):
            backbone = "vit"

        plot_gtVSerror_dir = (
            f"/s/red/a/nobackup/vision/anju/affine_tranformation/objects/"
            f"DINO/regressor/Linear_Regression/regression_analysis/"
            f"regression_comparison_plots/{model}"
        )
        os.makedirs(plot_gtVSerror_dir, exist_ok=True)

        parent_dir = (
            f"/s/red/a/nobackup/vision/anju/affine_tranformation/objects/"
            f"DINO/regressor/Linear_Regression/output/{model}/{category}/{subshape}/"
            f"{angle}/{scaleBlendedDir}/{subset}/{backbone}/{qs_type}/{test_size}/"
            f"{scale_str}/{model_type}/{subdir1}"
        )

        file_path = os.path.join(
            parent_dir,
            "vision_tower_Degrees_labels_and_predictions.csv"
        )

        if not os.path.exists(file_path):
            print(f"Missing file for {model}: {file_path}")
            continue

        df = pd.read_csv(file_path)
        df_all = df.iloc[:-3]

        # x_values = df_all.iloc[:, 0].values   # Ground Truth Angle
        # y_values = df_all.iloc[:, 2].values   # Absolute Error
        x_values = pd.to_numeric(df_all.iloc[:, 0], errors='coerce').values
        y_values = pd.to_numeric(df_all.iloc[:, 2], errors='coerce').values

        x_values_for_ticks = x_values  # store for ticks

        ax.plot(
            x_values,
            y_values,
            label=model,
            marker=markers[midx % len(markers)],
            linestyle='-',
            color=colors[midx % len(colors)],
            markersize=4,
            linewidth=1.6
        )

    # -----------------------------------
    # Skip if no valid data
    # -----------------------------------
    if x_values_for_ticks is None:
        print(f"No valid data for {subshape}, skipping...")
        plt.close(fig)
        continue

    # -----------------------------------
    # AXIS FORMATTING
    # -----------------------------------
    ax.set_title(f"{subshape}", fontsize=16, fontweight='bold')
    ax.set_xlabel('Ground Truth Angle', fontsize=14)
    ax.set_ylabel('Abs(Actual - Predicted) in Degrees', fontsize=14)

    ax.grid(True, axis='y', linestyle='--', linewidth=0.5)

    # -----------------------------------
    # DYNAMIC X-TICKS (ALIGNED TO DATA)
    # -----------------------------------
    step = 30
    max_angle = int(x_values_for_ticks.max())

    xticks = list(range(0, max_angle + 1, step))

    # Ensure last value is included
    if xticks[-1] != max_angle:
        xticks.append(max_angle)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)

    # Optional vertical guide lines
    for x in xticks:
        ax.axvline(
            x=x,
            color='gray',
            linestyle='--',
            linewidth=0.5,
            alpha=0.4,
            zorder=0
        )

    ax.tick_params(axis='x', labelsize=12, rotation=45)
    ax.tick_params(axis='y', labelsize=12)

    ax.legend(fontsize=12, loc='upper left', frameon=True)

    # -----------------------------------
    # SAVE FIGURE
    # -----------------------------------
    savepath = os.path.join(
        plot_gtVSerror_dir,
        f"{subshape}_regression_{model}.png"
    )

    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close(fig)

print("Done.")