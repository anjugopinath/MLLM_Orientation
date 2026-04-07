import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from utils import *
from datetime import datetime

def write_weight_analysis_info(file_path, layername, weights_sin, weights_cos, plot_dir):
    """
    Writes detailed weight analysis information to a file and generates a histogram.
    """
    # Create the directory for plots if it doesn't exist
    weight_plot_dir = os.path.join(plot_dir, 'Weight_Analysis_Histograms')
    os.makedirs(weight_plot_dir, exist_ok=True)

    # Create the file if it does not exist
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            pass  # Just create the file

    with open(file_path, "a") as f: # Changed to "a" (append) mode to add new analysis
        # Convert embedding_file to string to avoid dtype mismatch
        layername_str = str(layername)  # Ensure it's a string

        # Write date and time before everything else
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n------------------Date and Time: {now}------------------\n")

        f.write("Layername:\n")
        f.write(layername_str + "\n")

        # Concatenate the weights for combined analysis
        weights = np.concatenate((weights_sin, weights_cos))
        f.write(f"Original combined weights shape: {weights.shape}\n")

        # Ensure weights are a 1D array for easier processing if they are 2D (e.g., (1, N))
        if weights.ndim > 1:
            weights = weights.flatten()
        f.write(f"Flattened combined weights shape: {weights.shape}\n")

        # 1. Calculate the L2 norm (magnitude) of the weights
        # L2 norm = sqrt(sum(w_i^2))
        l2_norm = np.linalg.norm(weights)  # This is equivalent to np.sqrt(np.sum(weights**2))
        f.write(f"L2 Norm of combined model weights: {l2_norm:.6f}\n")

        if l2_norm == 0:
            f.write("L2 norm is zero. Cannot scale weights.\n")
            scaled_weights = np.zeros_like(weights)  # All scaled weights will be zero
        else:
            # 2. Scale the weights by dividing by the L2 norm
            scaled_weights = weights / l2_norm
            f.write(f"Scaled weights (first 10 elements): {scaled_weights[:10]}\n")
            f.write(f"Scaled weights (last 10 elements): {scaled_weights[-10:]}\n")

        # 3. Check how many of them are close to zero after scaling
        # Define a small tolerance for "close to zero"
        zero_tolerance_limits = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        num_zeros_list = []
        total_weights = len(scaled_weights)
        f.write(f"Total number of weights: {total_weights}\n")
        for index, zero_tolerance in enumerate(zero_tolerance_limits):
            num_zeros_after_scaling = np.sum(np.abs(scaled_weights) < zero_tolerance)
            num_zeros_list.append(num_zeros_after_scaling)
            percentage_zeros = (num_zeros_after_scaling / total_weights) * 100 if total_weights > 0 else 0
            f.write(f"Number of scaled weights close to zero (tolerance < {zero_tolerance}): {num_zeros_after_scaling}\n")
            f.write(f"With zero tolerance of {zero_tolerance}, percentage of scaled weights close to zero: {percentage_zeros:.2f}%\n")
            if(index>=1):
                f.write(f"Number of scaled weights between zero tolerances {zero_tolerance_limits[index]} and {zero_tolerance_limits[index-1]} is : {abs(num_zeros_list[index] - num_zeros_list[index-1])}\n")
                f.write("***********\n")

        # --- Grouping into buckets and generating histogram ---
        f.write("\n--- Scaled Weight Distribution (Buckets) ---\n")

        # Determine the number of bins. You can adjust this value.
        # A common heuristic is sqrt(n) or log2(n) + 1 (Sturges' formula)
        num_bins = int(np.sqrt(len(scaled_weights))) if len(scaled_weights) > 0 else 10
        if num_bins < 5: num_bins = 5 # Ensure a minimum number of bins

        # Calculate histogram: counts are the number of values in each bin, bin_edges define the bins
        counts, bin_edges = np.histogram(scaled_weights, bins=num_bins)

        f.write("Scaled Weight Buckets and Counts:\n")
        for i in range(len(counts)):
            lower_bound = bin_edges[i]
            upper_bound = bin_edges[i+1]
            f.write(f"  [{lower_bound:.6f}, {upper_bound:.6f}): {counts[i]} weights\n")

        # Generate and save the histogram plot
        fig = plt.figure(figsize=(10, 6))
        plt.hist(scaled_weights, bins=num_bins, edgecolor='black', alpha=0.7)
        plt.title(f'Histogram of Scaled Weights for {layername}')
        plt.xlabel('Scaled Weight Value')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()

        histogram_filename = f'scaled_weights_histogram_{layername}.png'
        histogram_filepath = os.path.join(weight_plot_dir, histogram_filename)
        plt.savefig(histogram_filepath)
        plt.close(fig)
        f.write(f"Histogram saved to: {histogram_filepath}\n")
        print(f"Histogram of scaled weights saved to {histogram_filepath}")

    print("--- End of Weight Analysis ---")

def write_weight_analysis_info2(file_path, layername, weights_sin, weights_cos):

    # Create the file if it does not exist
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            pass  # Just create the file
            
    with open(file_path, "r+") as f:
        content = f.read()
        # Convert embedding_file to string to avoid dtype mismatch
        layername_str = str(layername)  # Ensure it's a string
        
        # Check if the embedding file name is already in the file
        if f"{layername_str}\n" not in content:
            # Move to the end of file to append
            f.seek(0, os.SEEK_END)

            # Write date and time before everything else
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n------------------Date and Time: {now}------------------\n")
            
            f.write("Layername:\n")
            f.write(layername_str + "\n")

            # Concatenate the weights for combined analysis
            weights = np.concatenate((weights_sin, weights_cos))
            f.write(f"Original combined weights shape: {weights.shape}\n")
            

            # Ensure weights are a 1D array for easier processing if they are 2D (e.g., (1, N))
            if weights.ndim > 1:
                weights = weights.flatten()
            f.write(f"Flattened combined weights shape: {weights.shape}\n")

            # 1. Calculate the L2 norm (magnitude) of the weights
            # L2 norm = sqrt(sum(w_i^2))
            l2_norm = np.linalg.norm(weights) # This is equivalent to np.sqrt(np.sum(weights**2))
            f.write(f"L2 Norm of combined model weights: {l2_norm:.6f}\n")

            if l2_norm == 0:
                f.write("L2 norm is zero. Cannot scale weights.\n")
                scaled_weights = np.zeros_like(weights) # All scaled weights will be zero
            else:
                # 2. Scale the weights by dividing by the L2 norm
                scaled_weights = weights / l2_norm
                f.write(f"Scaled weights (first 10 elements): {scaled_weights[:10]}\n")
                f.write(f"Scaled weights (last 10 elements): {scaled_weights[-10:]}\n")

            # 3. Check how many of them are close to zero after scaling
            # Define a small tolerance for "close to zero"
            zero_tolerance = 1e-9
            num_zeros_after_scaling = np.sum(np.abs(scaled_weights) < zero_tolerance)
            total_weights = len(scaled_weights)
            percentage_zeros = (num_zeros_after_scaling / total_weights) * 100 if total_weights > 0 else 0

            f.write(f"Number of scaled weights close to zero (tolerance < {zero_tolerance}): {num_zeros_after_scaling}\n")
            f.write(f"Total number of weights: {total_weights}\n")
            f.write(f"Percentage of scaled weights close to zero: {percentage_zeros:.2f}%\n")      

def align(y_true, y_pred):
    """ Add or remove 2*pi to predicted angle to minimize difference from GT"""
    y_pred = y_pred.copy()
    y_pred[y_true-y_pred >  np.pi] += np.pi*2
    y_pred[y_true-y_pred < -np.pi] -= np.pi*2
    return y_pred

# Testing Phase for Ridge Regression
def test_model(model, test_loader, layername, output_dir, plot_dir):

    # Ensure PyTorch models are in evaluation mode
    if hasattr(model, "eval"):
        model.eval()
        
    all_labels_degrees = []

    # all_sin_labels = []
    # all_cos_labels = []
    all_sin_predictions_radians = []
    all_cos_predictions_radians = []

    # Loop through the test loader to collect embeddings and labels
    for embeddings, labels in test_loader:
        # Flatten embeddings for Ridge Regression
        embeddings = embeddings.view(-1).numpy()
        labels_degrees = labels.numpy().flatten()  # Ensure labels are flattened

        # Predict using the Ridge Regression model
        # predictions = model.predict(embeddings.reshape(1, -1))  # Reshape for a single sample
        sin_predictions, cos_predictions = model.predict(embeddings.reshape(1, -1))

        # Collect predictions and labels for metrics computation
        #COMPARE ANGLE IN RADIANS
        # all_sin_labels.append(np.sin(np.radians(labels)))
        # all_cos_labels.append(np.cos(np.radians(labels)))
        all_sin_predictions_radians.append(sin_predictions)
        all_cos_predictions_radians.append(cos_predictions)
    
        all_labels_degrees.append(labels_degrees)

        # #COMPARE ANGLE IN DEGREES
        # all_sin_labels.append(np.sin(labels))
        # all_cos_labels.append(np.cos(labels))
        # all_sin_predictions.append(np.degrees(sin_predictions))
        # all_cos_predictions.append(np.degrees(cos_predictions))
        

    # # Before concatenation, print the size of the lists
    # print(f"Size of all_labels before concatenation: {len(all_labels)}")
    # print(f"Size of all_predictions before concatenation: {len(all_predictions)}")

    all_labels_degrees = np.concatenate(all_labels_degrees, axis=0)
    # Flatten the lists for MAE and R² computation
    # all_sin_labels = np.concatenate(all_sin_labels, axis=0)
    # all_cos_labels = np.concatenate(all_cos_labels, axis=0)
    all_sin_predictions_radians = np.concatenate(all_sin_predictions_radians, axis=0)
    all_cos_predictions_radians = np.concatenate(all_cos_predictions_radians, axis=0)

    # # After concatenation, print the new size
    # print(f"Size of all_labels after concatenation: {all_labels.shape}")
    # print(f"Size of all_predictions after concatenation: {all_predictions.shape}")

    # print("all_labels : ", all_labels)
    # print("all_predictions : ", all_predictions)

    # all_sin_predictions_aligned = align(all_sin_labels, all_sin_predictions)
    # all_cos_predictions_aligned = align(all_cos_labels, all_cos_predictions)
    all_labels_radians = np.radians(all_labels_degrees)
    all_predictions_aligned_radians = align(all_labels_radians, np.arctan2(all_sin_predictions_radians, all_cos_predictions_radians))

    # # Calculate MAE and R²
    # mae_sin = mean_absolute_error(all_sin_labels, all_sin_predictions_aligned)
    # r2_sin = r2_score(all_sin_labels, all_sin_predictions_aligned)
    # print(f'Mean Absolute Error on Test Set (Sine): {mae_sin:.2f} degrees')
    # print(f'R^2 Score on Test Set (Sine): {r2_sin:.2f}')
    # # Calculate MAE and R²
    # mae_cos = mean_absolute_error(all_cos_labels, all_cos_predictions_aligned)
    # r2_cos = r2_score(all_cos_labels, all_cos_predictions_aligned)
    # print(f'Mean Absolute Error on Test Set (Cosine): {mae_cos:.2f} degrees')
    # print(f'R^2 Score on Test Set (Cosine): {r2_cos:.2f}')

    #CALCULATIONS IN RADIANS
    # Calculate MAE and R²
    # mae = mean_absolute_error(all_labels_radians, all_predictions_aligned_radians)
    # r2 = r2_score(all_labels_radians, all_predictions_aligned_radians)
    # sum_absolute_errors = np.sum(np.abs(all_labels_radians - all_predictions_aligned_radians))

    # print("Sum of Absolute Errors:", sum_absolute_errors)
    # print(f'Mean Absolute Error on Test Set: {mae:.2f} radians')
    # print(f'R^2 Score on Test Set: {r2:.2f}')
    # write_labels_to_csv(all_labels_radians, all_predictions_aligned_radians, layername, mae, sum_absolute_errors, r2, output_dir, "Radians")
    # Write predictions and labels to an Excel file
    # write_labels_to_csv(all_sin_labels, all_sin_predictions_aligned, layername, mae_sin, r2_sin, output_dir, "Sine")
    # write_labels_to_csv(all_cos_labels, all_cos_predictions_aligned, layername, mae_cos, r2_cos, output_dir, "Cosine")
    
    #CALCULATIONS IN DEGREES
    # Calculate MAE and R²
    all_predictions_aligned_degrees = np.degrees(all_predictions_aligned_radians)
    mae = mean_absolute_error(all_labels_degrees, all_predictions_aligned_degrees)
    r2 = r2_score(all_labels_degrees, all_predictions_aligned_degrees)
    sum_absolute_errors = np.sum(np.abs(all_labels_degrees - all_predictions_aligned_degrees))

    print("Sum of Absolute Errors:", sum_absolute_errors)
    print(f'Mean Absolute Error on Test Set: {mae:.2f} radians')
    print(f'R^2 Score on Test Set: {r2:.2f}')
    write_labels_to_csv(all_labels_degrees, all_predictions_aligned_degrees, layername, mae, sum_absolute_errors, r2, output_dir, "Degrees")

    #CALCULATIONS IN DEGREES UNALIGNED
    # all_predictions_unaligned_degrees = np.degrees(np.arctan2(all_sin_predictions_radians, all_cos_predictions_radians))
    # mae = mean_absolute_error(all_labels_degrees, all_predictions_unaligned_degrees)
    # r2 = r2_score(all_labels_degrees, all_predictions_unaligned_degrees)
    # sum_absolute_errors = np.sum(np.abs(all_labels_degrees - all_predictions_unaligned_degrees))

    # print("Sum of Absolute Errors:", sum_absolute_errors)
    # print(f'Mean Absolute Error on Test Set: {mae:.2f} degrees unaligned')
    # print(f'R^2 Score on Test Set: {r2:.2f}')
    # write_labels_to_csv(all_labels_degrees, all_predictions_unaligned_degrees, layername, mae, sum_absolute_errors, r2, output_dir, "Degrees Unligned")
    # plot_actual_predicted(all_labels, all_predictions)
    # plot_sine_cosine(np.arctan2(all_sin_predictions_aligned, all_cos_predictions_aligned), np.radians(all_labels), layername, "degrees", plot_dir)
    # plot_sine_cosine(np.arctan2(np.radians(all_sin_predictions_aligned), np.radians(all_cos_predictions_aligned)), all_labels, layername, "radians", plot_dir)
    plot_sine_cosine_dir = os.path.join(plot_dir, 'Sine-Cosine_Model')
    os.makedirs(plot_sine_cosine_dir, exist_ok=True)
    plot_sine_cosine(all_predictions_aligned_degrees, all_labels_degrees, layername, plot_sine_cosine_dir, "Aligned", "Degrees")
    # plot_sine_cosine(all_predictions_aligned_radians, all_labels_radians, layername, plot_sine_cosine_dir, "Aligned", "Radians")

    # plot_sine_cosine(all_predictions_unaligned_degrees, all_labels_degrees, layername, plot_sine_cosine_dir, "Unaligned", "Degrees")

    # --- Start of Weight Analysis ---
    print("\n--- Linear Regression Model Weight Analysis ---")
    # Check if the model has the internal Ridge models and their 'coef_' attributes
    if hasattr(model, 'model_sin_ridge') and hasattr(model.model_sin_ridge, 'coef_') and \
       hasattr(model, 'model_cos_ridge') and hasattr(model.model_cos_ridge, 'coef_'):

        # Get coefficients from both sine and cosine models
        weights_sin = model.model_sin_ridge.coef_
        weights_cos = model.model_cos_ridge.coef_

        weight_analysis_file_path = f"{output_dir}/weight_analysis_model_test.txt"
        write_weight_analysis_info(weight_analysis_file_path, layername, weights_sin, weights_cos, plot_dir)

    else:
        print("Model does not have the expected 'model_sin_ridge' or 'model_cos_ridge' attributes with 'coef_'. Cannot perform weight analysis.")
    print("--- End of Weight Analysis ---")
    
    