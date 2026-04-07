import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def align(y_true, y_pred):
    """Add or remove 2*pi to predicted angle to minimize difference from GT"""
    y_pred = y_pred.copy()
    y_pred[y_true - y_pred > np.pi] += np.pi*2
    y_pred[y_true - y_pred < -np.pi] -= np.pi*2
    return y_pred

def main(other_nums, anchor, modelnames, blended_shape, origScales, scaleTitles,
         patch_analysis_modes, filename, category, subdir1, subdir2, backbone, make_compact):

    # --- Map blend name to readable form ---
    str_shape = blended_shape

    for modelname in modelnames:
        if modelname == "llava-ov-qwen2-7b":
            str_modelname = "llavaOV"
            if(category=="natural_cropped" or category=="in_place_rotated"):
                num_bins_total = 3359232
                step = [8000, 16000, 128000, 270000, 540000, 810000, 1080000, 1620000, 1890000, 2160000, 3000000]  
            parent_path = f"/s/red/a/nobackup/vision/anju/affine_tranformation/objects/DINO/regressor/Linear_Regression/output/llava-ov-qwen2-7b/{category}/{blended_shape}/1_degrees_FIXED"
        elif modelname == "Qwen2.5-VL-7B-Instruct":
            str_modelname = "Qwen2.5-VL-7B-Instruct"
            if(category=="natural_cropped" or category=="in_place_rotated"):
                if(blended_shape in ["fish","indoor","train"]):
                    num_bins_total = 501760
                    step = [2000, 5000, 8000, 16000, 25000, 75000, 128000, 270000, 335000, 500000]
                else:
                    num_bins_total = 829440
                    step = [2000, 8000, 16000, 25000, 75000, 128000, 270000, 335000, 500000, 800000]
            parent_path = f"/s/red/a/nobackup/vision/anju/affine_tranformation/objects/DINO/regressor/Linear_Regression/output/Qwen2.5-VL-7B-Instruct/{category}/{blended_shape}/1_degrees_FIXED" 
        else:
            print(f"Error: Unknown modelname: {modelname}")
            continue

        # --- Figure Size & Font Configuration (Research Paper Ready) ---
        base_width = 5.2 if make_compact else 6
        base_height = 2.6  # smaller per-row height for compact figure

        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 13,
            'axes.labelsize': 12,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 10,
            'lines.linewidth': 1.4,
            'lines.markersize': 4,
        })

        # Create subplot grid: rows = scales, cols = patch_modes
        fig, axes = plt.subplots(
            nrows=len(origScales),
            ncols=len(patch_analysis_modes),
            figsize=(base_width * len(patch_analysis_modes),
                     base_height * len(origScales)),
            squeeze=False,
            sharex=True
        )

        for i_scale, scale in enumerate(origScales):
            for j_mode, patch_analysis_mode in enumerate(patch_analysis_modes):
                ax = axes[i_scale][j_mode]

                # Patch analysis naming
                if patch_analysis_mode == "byModelWeight":
                    str_mode = "modelweight"
                elif patch_analysis_mode == "byAbsDiff":
                    str_mode = "absdiff"
                elif patch_analysis_mode == "byRandomLocations":
                    str_mode = "random"
                else:
                    print(f"Error: Unknown patch_analysis_mode: {patch_analysis_mode}")
                    continue

                # Axis format
                ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
                ax.tick_params(axis='x', rotation=0)

                # Hide x labels except on bottom row
                if i_scale < len(origScales) - 1:
                    ax.tick_params(axis='x', labelbottom=False)

                # Scale label inside plot
                ax.text(
                    0.5, 0.92,
                    f"{scaleTitles[i_scale]}",
                    transform=ax.transAxes,
                    fontsize=12,
                    fontweight='bold',
                    ha='center',
                    va='top'
                )

                print(f"\n=== Model: {modelname}, Scale: {scale}, Patch Mode: {patch_analysis_mode} ===")

                all_bins_total = []

                for num in other_nums:
                    if num == anchor:
                        continue

                    all_diffs, all_bins = [], []
                    pair_str = f"{anchor}into{num}"

                    for num_bins in step:
                        if num_bins == 0:
                            scaleBlendedDir = scale
                        else:
                            if patch_analysis_mode == "byModelWeight":
                                scaleBlendedDir = f"{scale}_{pair_str}_{num_bins}modelweight"
                            elif patch_analysis_mode == "byAbsDiff":
                                scaleBlendedDir = f"{scale}_{pair_str}_{num_bins}absdiff"
                            elif patch_analysis_mode == "byRandomLocations":
                                scaleBlendedDir = f"{scale}_{pair_str}_{num_bins}random"
                        subdir_path = f"{subdir1}/{backbone}/what_angle/36_test_samples/Scaled/RidgeRegression/{subdir2}"
                        
                        input_path = os.path.join(parent_path,scaleBlendedDir,subdir_path, filename)

                        if not os.path.exists(input_path):
                            print(f"⚠️ Missing file: {input_path}")
                            continue

                        try:
                            df = pd.read_csv(input_path)
                        except pd.errors.EmptyDataError:
                            print(f"⚠️ Empty CSV file: {input_path}")
                            continue

                        df["Actual"] = pd.to_numeric(df["Actual"], errors="coerce")
                        df = df.dropna(subset=["Actual"])
                        df_filtered = df[df["Actual"] == num]

                        if df_filtered.empty:
                            print(f"⚠️ Empty filtered data for Actual={num} at {scaleBlendedDir}")
                            continue

                        predicted_vals = df_filtered["Predicted"].astype(float)
                        actual_vals = df_filtered["Actual"].astype(float)

                        # Align predictions for circular angles
                        y_true = np.deg2rad(np.full_like(predicted_vals, anchor))
                        y_pred = np.deg2rad(predicted_vals)
                        y_pred_aligned = align(y_true, y_pred)
                        pred_deg_aligned = np.degrees(y_pred_aligned)

                        # Compute normalized difference
                        normalized_diff = np.abs((pred_deg_aligned - anchor) / (actual_vals - anchor))
                        avg_normalized_diff = normalized_diff.mean()

                        all_diffs.append(avg_normalized_diff)
                        all_bins.append(num_bins)
                        all_bins_total.append(num_bins)

                        print(f"  NumBins: {num_bins:<6} | Actual: {num:<3} | Avg Norm Diff: {avg_normalized_diff:.4f}")

                    if all_diffs:
                        ax.plot(all_bins, all_diffs, marker="o", label=f"Actual={num}")

                # Vertical guide lines
                unique_bins = sorted(set(all_bins_total))
                ax.set_ylim(top=ax.get_ylim()[1] * 1.05)
                y_max_current = ax.get_ylim()[1]

                for x in unique_bins:
                    ax.axvline(x=x, color='gray', linestyle='--', linewidth=0.6, alpha=0.7)
                    # if i_scale == len(origScales) - 1:
                    # if i_scale == 0:
                    #     ax.text(
                    #         x, y_max_current, f"{int(x)}",
                    #         ha='left', va='bottom',
                    #         rotation=45,
                    #         fontsize=7.5, color='black', alpha=0.8,
                    #         clip_on=False
                    #     )
                    if i_scale == 0:
                        # Adjust font size for first two bins (8000, 16000) in llava1.6 dog_on_beach
                        if modelname == "llava-ov-qwen2-7b": #and blended_shape == "dog_on_beach" and x in [8000,16000]:
                           
                            if x == 8000:
                                x_offset = 70000
                            else:
                                x_offset = -7000
                        else:
                            if(blended_shape in ["fish","indoor","train"]):
                                if x==2000:
                                    x_offset = 6500
                                elif x==5000:
                                    x_offset = 2500
                                elif x==8000:
                                    x_offset = 0
                                elif x==16000:
                                    x_offset = 0
                                elif x==25000:
                                    x_offset = 0
                                else:
                                    x_offset = 0    
                            
                            else:
                                if x==2000:
                                    x_offset = 10000
                                elif x==5000:
                                    x_offset = 8000
                                elif x==8000:
                                    x_offset = 6000
                                elif x==16000:
                                    x_offset = 0
                                elif x==25000:
                                    x_offset = -3000
                                else:
                                    x_offset = 0                            
                        fontsize = 3
                        alpha = 0.8
                        ax.text(
                            x-x_offset, y_max_current, f"{int(x)}",
                            ha='left', va='bottom',
                            rotation=45,
                            fontsize=fontsize,
                            color='black',
                            alpha=alpha,
                            clip_on=False
                        )

                # Reference line
                ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
                if i_scale == len(origScales) - 1:
                    ax.set_xlabel("Number of Replaced Features")

                # Remove per-plot y-labels (we’ll add a shared one later)
                # ax.set_ylabel("|(Predicted - 3°) / (Actual - 3°)|")

                ax.grid(True, linestyle='--', alpha=0.5, axis='y')
                if i_scale == 0:
                    ax.legend(title="Target Angle", fontsize=5, loc='best')

        # --- Layout and shared labels ---
        plt.tight_layout(pad=1.5)
        plt.subplots_adjust(hspace=0.15, left=0.12)

        # One central shared y-axis label
        fig.text(
            0.02, 0.5,
            "|(Predicted - 9°) / (Actual - 9°)|",
            va='center',
            rotation='vertical',
            fontsize=8,
            fontweight='bold'
        )
        # Bigger title line
        # fig.text(
        #     0.012, 0.6,
        #     "|(Predicted - 9°) / (Actual - 9°)|",
        #     va='center',
        #     rotation='vertical',
        #     fontsize=8,   # bigger
        #     fontweight='bold'
        # )

        # # Smaller explanatory text
        # fig.text(
        #     0.032, 0.6,
        #     "y = 1 (predicted matches target orientation)\n"
        #     "y = 0 (predicted matches anchor orientation)",
        #     va='center',
        #     rotation='vertical',
        #     fontsize=4,   # smaller
        #     fontweight='bold'
        # )

        # Save high-resolution output
        output_dir = f"feature_substitution_plots/{str_modelname}/{category}"
        os.makedirs(output_dir, exist_ok=True)
        save_name = f"{output_dir}/{str_shape}_{str_mode}.png"
        plt.savefig(save_name, dpi=400, bbox_inches='tight')
        print(f"\n✅ Saved plot: {save_name}")
        plt.close(fig)
