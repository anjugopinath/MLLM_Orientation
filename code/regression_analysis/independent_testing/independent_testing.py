import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import sys
sys.path.append("/<dirname>/<name>/affine_tranformation/objects/DINO/regressor/Linear_Regression")
import model_ridge_regression  # must import before unpickling

# --------------------------------------
# Align function (handles wrap-around)
# --------------------------------------
def align(y_true, y_pred):
    """Add or remove 2π to predicted angle to minimize difference from GT"""
    y_pred = y_pred.copy()
    y_pred[y_true - y_pred >  np.pi] += np.pi * 2
    y_pred[y_true - y_pred < -np.pi] -= np.pi * 2
    return y_pred

# --------------------------------------
# Paths and settings
# --------------------------------------
test_csv = "./vision_model.post_layernorm_test_dataset.csv"

#backgrounds (synthetic) -> chessboard, grid_lines, horizontal_lines, vertical_lines
#backgrounds (real) -> beach
shape_foreground = "dog"
shape_background_training = "grid_lines"
shape_background_testing = "vertical_lines"
symmetrical_background = False

background_rot_type_training = "_bRot_fStat" # options: "", "_b.5Rot_f.5Rot_1degreeTotal", "_bRot_fStat"
shape_training = f"{shape_foreground}_on_{shape_background_training}{background_rot_type_training}"

background_rot_type_testing = "_bRot_fStat"  # options: "", "_b.5Rot_f.5Rot_1degreeTotal", "_bRot_fStat"
shape_testing = f"{shape_foreground}_on_{shape_background_testing}{background_rot_type_testing}"
if("chessboard" not in shape_testing and "bRot_fStat" in shape_testing):
    symmetrical_background = True
if("beach" in shape_testing):
    symmetrical_background = False

model_name = "llava-v1.5-13b" #Options - llava-v1.5-13b, llava-v1.6-vicuna-13b
scale = "scale1"  # options: scale1, scale4, scale16

out_csv = f"output/independent_test_predictions_{model_name}_{scale}_train--{shape_training}_test--{shape_testing}.csv"
out_plot_path = f"output/independent_test_predictions_{model_name}_{scale}_train--{shape_training}_test--{shape_testing}.png"

if(model_name == "llava-v1.6-vicuna-13b"):
    embeddings_path = f"/<dirname>/<name>/g-t_embedding_classifier/llava1.6/LLaVA/llava/eval/pca_images/what_angle/1_degree_FIXED/{shape_testing}/{scale}/embeddings/vision_model.encoder.layers.23.layer_norm2.npy"

elif(model_name == "llava-v1.5-13b"):
    embeddings_path = f"/<dirname>/<name>/g-t_embedding_classifier/llava/LLaVA/llava/eval/pca_images/what_angle/1_degree_FIXED/{shape_testing}/{scale}/embeddings/vision_model.encoder.layers.23.layer_norm2.npy"

saved_model_path = f"/<dirname>/<name>/affine_tranformation/objects/DINO/regressor/Linear_Regression/output/{model_name}/blended/{shape_training}/1_degrees_FIXED/{scale}/_visionEncAll/clip_vitl336/what_angle/72_test_samples/Scaled/RidgeRegression/train_and_test/saved_models/vision_model.encoder.layers.23.layer_norm2_ridge_model.pkl"
# --------------------------------------
# Load CSV and labels
# --------------------------------------
df = pd.read_csv(test_csv)
labels_degrees = df["label"].values.astype(float)
print(f"Loaded {len(labels_degrees)} labels from {test_csv}")
print(f"labels (degrees): {labels_degrees}")

# Convert test indices to match embedding keys
test_indices = [str(int(v)).replace(".0", "") for v in labels_degrees]
# print(f"Test indices: {test_indices}")
# --------------------------------------
# Optionally filter out >180° if background is symmetrical
# --------------------------------------
if "symmetrical_background" in locals() and symmetrical_background:
    print("Symmetrical background detected — keeping only samples with label < 180°")
    mask = labels_degrees < 180
    labels_degrees = labels_degrees[mask]
    test_indices = [t for t, keep in zip(test_indices, mask) if keep]
    print(f"Filtered to {len(labels_degrees)} samples (labels < 180°)")

# sys.out(1)

# --------------------------------------
# Load embedding dictionary
# --------------------------------------
ref_emb = np.load(embeddings_path, allow_pickle=True).item()

# # Build X_test array
# X_test = []
# for idx in test_indices:
#     if idx not in ref_emb:
#         raise KeyError(f"Embedding key '{idx}' not found in ref_emb")
#     X_test.append(ref_emb[idx].flatten())
# X_test = np.array(X_test)
# print(f"Loaded {len(X_test)} embeddings, each with shape {X_test[0].shape}")
# --------------------------------------
# Build X_test array (convert torch.Tensor → np.ndarray safely)
# --------------------------------------
X_test = []
for idx in test_indices:
    if idx not in ref_emb:
        raise KeyError(f"Embedding key '{idx}' not found in ref_emb")
    
    emb = ref_emb[idx]
    # --- Convert PyTorch tensor to NumPy ---
    if hasattr(emb, "detach"):
        emb = emb.detach().cpu().numpy()
    elif not isinstance(emb, np.ndarray):
        emb = np.array(emb)

    X_test.append(emb.flatten())

# Use np.stack to enforce consistent shape
X_test = np.stack(X_test)

print(f"Loaded {len(X_test)} embeddings, each with shape {X_test[0].shape}, dtype={X_test.dtype}")


# --------------------------------------
# Load saved Ridge model
# --------------------------------------
print(f"Loading saved model from {saved_model_path}...")
with open(saved_model_path, "rb") as f:
    model = pickle.load(f)

# --------------------------------------
# Predict sin/cos and convert to radians
# --------------------------------------
if isinstance(model, tuple) and len(model) == 2:
    model_sin, model_cos = model
    pred_sin = model_sin.predict(X_test)
    pred_cos = model_cos.predict(X_test)
    preds = np.stack([pred_sin, pred_cos], axis=1)
else:
    preds = model.predict(X_test)
    if isinstance(preds, tuple):
        preds = np.stack(preds, axis=1)

pred_sin = preds[:, 0]
pred_cos = preds[:, 1]

pred_radians = np.arctan2(pred_sin, pred_cos)
labels_radians = np.radians(labels_degrees)

# Align predictions
pred_radians_aligned = align(labels_radians, pred_radians)
pred_degrees_aligned = np.degrees(pred_radians_aligned)

# --------------------------------------
# Evaluate
# --------------------------------------
mae = mean_absolute_error(labels_degrees, pred_degrees_aligned)
r2 = r2_score(labels_degrees, pred_degrees_aligned)
sum_abs_err = np.sum(np.abs(labels_degrees - pred_degrees_aligned))

print("\n=== Test Results (Aligned) ===")
print(f"Sum of Absolute Errors: {sum_abs_err:.4f}")
print(f"Mean Absolute Error (Degrees): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

for i, (gt, pr) in enumerate(zip(labels_degrees, pred_degrees_aligned)):
    print(f"Sample {i}: GT={gt:.2f}°, Pred={pr:.2f}°")

# --------------------------------------
# Save predictions sorted by true_deg
# --------------------------------------
df_out = pd.DataFrame({
    "true_deg": labels_degrees,
    "pred_deg_aligned": pred_degrees_aligned
})

# Compute and add difference column
df_out["true_minus_pred"] = df_out["true_deg"] - df_out["pred_deg_aligned"]

df_out = df_out.sort_values(by="true_deg").reset_index(drop=True)
df_out.to_csv(out_csv, index=False)
# --- Append summary stats to the same CSV ---
with open(out_csv, "a") as f:
    f.write("\n")  # blank line separator
    f.write(f"# SumAbsErr,{sum_abs_err:.6f}\n")
    f.write(f"# MAE,{mae:.6f}\n")
    f.write(f"# R2,{r2:.6f}\n")

print(f"\nSaved aligned predictions + summary stats to {out_csv}")

# --------------------------------------
# Plot GT vs Predicted and Error vs True Angle
# --------------------------------------


x = df_out["true_deg"]
y_pred = df_out["pred_deg_aligned"]
y_err = df_out["true_minus_pred"]

# --- Font Size Configuration ---
TITLE_FONT_SIZE = 16
LABEL_FONT_SIZE = 14
TICK_FONT_SIZE = 12
# --------------------------------------------------------------------------

plt.figure(figsize=(5, 5))

plt.scatter(x, y_pred, alpha=0.7, label="Predictions")
plt.plot(x, y_pred, color='blue', alpha=0.5, linewidth=1, label="Trend")
plt.plot([0, 360], [0, 360], 'r--', label="Ideal")

# Make Title and Axis Labels Bold and Increase Font Size
plt.xlabel("True Angle (deg)", fontweight='bold', fontsize=LABEL_FONT_SIZE)
plt.ylabel("Predicted Angle (deg)", fontweight='bold', fontsize=LABEL_FONT_SIZE)
plt.title("Predictions vs Ground Truth", fontweight='bold', fontsize=TITLE_FONT_SIZE)

# Increase font size AND set font weight for tick labels
plt.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)

# Get the tick labels and set their font weight to bold
for tick in plt.gca().get_xticklabels():
    tick.set_fontweight('bold')
for tick in plt.gca().get_yticklabels():
    tick.set_fontweight('bold')

plt.grid(True, linestyle="--", alpha=0.5)

# Generate Legend and Increase Font Size
legend = plt.legend(fontsize=LABEL_FONT_SIZE)

# Make Legend Text Bold
for text in legend.get_texts():
    text.set_fontweight('bold')

# Tighten layout and remove extra margins
# Note: Using plt.tight_layout and plt.savefig(bbox_inches='tight') together 
# is the most robust way to eliminate excess white space.
plt.tight_layout(pad=0.2)
# Removed manual subplots_adjust as tight_layout often handles it better
# plt.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.1) 

# Save without whitespace around figure
plt.savefig(out_plot_path, bbox_inches='tight', pad_inches=0.05, dpi=300)

# Display the plot
plt.show()

# # --- Subplot 2: Error vs True Angle ---
# plt.subplot(1, 2, 2)
# plt.scatter(x, y_err, alpha=0.7, color='orange', label="Error (true - pred)")
# plt.axhline(0, color='red', linestyle='--', linewidth=1)
# plt.xlabel("True Angle (deg)")
# plt.ylabel("True - Pred (deg)")
# plt.title("Prediction Error vs True Angle")
# plt.grid(True, linestyle="--", alpha=0.5)
# plt.legend()

# plt.tight_layout()
# plt.savefig(out_plot_path)
# plt.close()
# print(f"Saved combined plot (GT vs Pred and Error vs True) to {out_plot_path}")

