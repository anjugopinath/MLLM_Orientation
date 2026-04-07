import os
import sys
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, chisquare, shapiro, cramervonmises, kstest, norm

def extract_sorted_dict_keys(data,num_samples):
    print("Inside extract_sorted_dict_keys -> num_samples : ", num_samples)
    if isinstance(data.item(), dict):
        data = data.item()
        print(data.keys())
        numeric_keys = sorted(data.keys(), key=lambda x: float(x))
        print("len keys : ", len(numeric_keys))
        first_val = list(data.values())[0]
        # print("Shape of the first val:", np.array(first_val).shape)
        # print("Type of the first val:", type(first_val))
        labels = numeric_keys[:num_samples]
    else:
        print("Data is not a dictionary. Here is the content:")
        print(data)
        labels = None
    
    return labels

def write_labels_to_csv(all_labels, all_predictions, layername, mae, sum_absolute_errors, r2, output_dir, angle_repr):

    # Assuming all_labels and all_predictions are already defined as lists or numpy arrays
    # If they are numpy arrays, convert them to lists:
    # Assuming all_labels and all_predictions are already lists
    all_labels = all_labels.tolist()
    all_predictions = all_predictions.tolist()

    # Combine labels and predictions, sort them numerically based on labels
    sorted_pairs = sorted(zip(all_labels, all_predictions), key=lambda x: x[0])

    # Unzip the sorted pairs
    sorted_labels, sorted_predictions = zip(*sorted_pairs)

    # Convert to strings
    sorted_labels = [str(label) for label in sorted_labels]
    sorted_predictions = [str(prediction) for prediction in sorted_predictions]

    # Create a DataFrame from the sorted lists
    df = pd.DataFrame({
        'Actual': pd.to_numeric(sorted_labels, errors='coerce'),
        'Predicted': pd.to_numeric(sorted_predictions, errors='coerce')
    })

    df[f'{angle_repr}_difference'] = (df['Actual'] - df['Predicted'])
    
    # Specify the output CSV file path
    output_csv_path = f'{output_dir}/{layername}_{angle_repr}_labels_and_predictions.csv'
    df.to_csv(output_csv_path, index=False)

    with open(output_csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        
        # Append the metrics as new rows
        writer.writerow([f"Mean Absolute Error on Test Set: {mae:.2f} {angle_repr}"])
        writer.writerow([f"Sum Absolute Error on Test Set: {sum_absolute_errors:.2f} {angle_repr}"])
        writer.writerow([f"R^2 Score on Test Set: {r2:.2f}"])
    print(f"Data has been written to {output_csv_path}")

    with open(f'{output_dir}/global_results.csv', mode="a", newline="") as file:
        writer = csv.writer(file)
        
        # Append the metrics as new rows
        # writer.writerow([f" ***** {layername} ***** "])
        writer.writerow([f" -- {angle_repr} MAE and R^2 Scores -- "])
        writer.writerow([])
        writer.writerow([f"Mean Absolute Error on Test Set: {mae:.2f} {angle_repr}"])
        writer.writerow([f"Sum Absolute Error on Test Set: {sum_absolute_errors:.2f} {angle_repr}"])
        writer.writerow([f"R^2 Score on Test Set: {r2:.2f}"])
        


    formatted_df = df.to_string(index=False, justify='right', col_space=15)
    print(formatted_df)


def save_dataset_to_csv(dataset, filename, subset):
    """
    Save image filenames and their labels to a CSV file.
    Args:
        dataset: The dataset object (must implement __getitem__).
        filename: The name of the CSV file.
    """
    image_names = []
    labels = []

    for idx in range(len(dataset)):
        # Get the filename and label for each sample
        _, label = dataset[idx]  # Extract values (ignoring image tensor and embedding)
        
        # Reconstruct the image filename
        label_str = str(int(label))  # Convert label to string
        # image_name = f"{dataset.shape}_{label_str}.png"
        
        # Append to the lists
        # image_names.append(image_name)
        labels.append(label)

    # Create a DataFrame with filenames and labels
    df = pd.DataFrame({
        # "image_name": image_names,
        "label": labels
    })

    # Save to CSV
    df.to_csv(filename, index=False)

def plot_actual_predicted(actual, predicted):

    # plt.scatter(actual, predicted, alpha=0.7, label='Predicted vs Actual')
    # plt.plot([min(actual), max(actual)], [min(actual), max(actual)], 'r--', label='Perfect Prediction')
    # plt.xlabel('Actual Rotation (degrees)')
    # plt.ylabel('Predicted Rotation (degrees)')
    # plt.legend()
    # plt.grid(True)

    # # Save the plot to a file
    # plt.savefig('scatter_actual_vs_predicted.png', dpi=300, bbox_inches='tight')  # Saves with high resolution
    # plt.close()  # Closes the plot to free memory

    fig = plt.figure(figsize=(8, 6))
    plt.scatter(actual, predicted, alpha=0.7, label='Predicted vs Actual')
    plt.plot([min(actual), max(actual)], [min(actual), max(actual)], 'r--', label='Perfect Prediction')

    # Add annotations for each point
    for i in range(len(actual)):
        plt.text(actual[i], predicted[i], f'[{actual[i]:.1f}, {predicted[i]:.1f}]', fontsize=8, alpha=0.7)

    plt.xlabel('Actual Rotation (degrees)')
    plt.ylabel('Predicted Rotation (degrees)')
    plt.legend()
    plt.grid(True)

    # Save the annotated plot
    plt.savefig('scatter_actual_vs_predicted_annotated.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_sine_cosine(y_trig_pred, y_actual, layername, plot_dir, alignment, angle_repr):

    fig = plt.figure(figsize=(8, 6))
    print("y_trig_pred size : ", y_trig_pred.size, "\ny_actual size : ",y_actual.size)
    plt.scatter(y_trig_pred, y_actual)
    plt.title(f'Sine-cosine model : {alignment}-{angle_repr}'); plt.xlabel('prediction'); plt.ylabel('actual')
    save_path = f'{plot_dir}/{layername}_{alignment}_{angle_repr}_Sine-Cosine_Model.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

# def perform_chi_squared_test(angle_diff, num_bins):

#     # 2. Generate observed frequencies (histogram)
#     observed_freq, bin_edges = np.histogram(angle_diff, bins=num_bins)

#     # 3. Fit a normal distribution to the data
#     mu, sigma = norm.fit(angle_diff)

#     # 4. Calculate expected frequencies
#     bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
#     expected_freq = norm.pdf(bin_centers, mu, sigma) * len(angle_diff) * (bin_edges[1] - bin_edges[0])

#     print("\nBEFORE NORMALIZATION - Observed vs. Expected Frequencies:")
#     for i in range(len(observed_freq)):
#         print(f"Bin {i+1}: Observed = {observed_freq[i]}, Expected = {expected_freq[i]:.4f}")

#     # 5. Normalize expected frequencies
#     expected_freq = expected_freq / np.sum(expected_freq) * np.sum(observed_freq)

#     # 6. Perform the chi-squared test
#     chi_stat, p_value = chisquare(f_obs=observed_freq, f_exp=expected_freq)

#     # Print results
#     print(f"Chi-squared statistic: {chi_stat}")
#     print(f"P-value: {p_value}")

#     # Print observed and expected frequencies side by side
#     print("\nAFTER NORMALIZATION - Observed vs. Expected Frequencies:")
#     for i in range(len(observed_freq)):
#         print(f"Bin {i+1}: Observed = {observed_freq[i]}, Expected = {expected_freq[i]:.4f}")

#     # Check if expected frequencies are too low.
#     if any(x < 5 for x in expected_freq):
#         print("Warning: Some expected frequencies are less than 5. Chi-squared test may not be reliable.")

# def group_by_dynamic_range(numbers):
#     numbers = sorted(numbers)
#     min_val, max_val = min(numbers), max(numbers)

#     if min_val == max_val:
#         return {f"{min_val}–{max_val}": numbers}

#     # Dynamic number of groups (e.g., sqrt of data size)
#     num_groups = max(1, int(np.sqrt(len(numbers))))
#     bins = np.linspace(min_val, max_val, num_groups + 1)

#     # Step 1: Create initial groups
#     raw_groups = [[] for _ in range(num_groups)]
#     for num in numbers:
#         for i in range(num_groups):
#             if bins[i] <= num < bins[i+1] or (i == num_groups - 1 and num == max_val):
#                 raw_groups[i].append(num)
#                 break

#     # Step 2: Merge small groups
#     merged_groups = []
#     i = 0
#     while i < len(raw_groups):
#         group = raw_groups[i]
#         if len(group) < 5:
#             # Try merging with the next group
#             if i + 1 < len(raw_groups):
#                 raw_groups[i + 1] = group + raw_groups[i + 1]
#             # Or merge with the previous group if it's the last one
#             elif i > 0:
#                 merged_groups[-1] += group
#             # If no other option, just keep it
#             else:
#                 merged_groups.append(group)
#         else:
#             merged_groups.append(group)
#         i += 1

#     # Step 3: Build new bin labels for merged groups
#     result = {}
#     current_bin = 0
#     for group in merged_groups:
#         if not group:
#             continue
#         start = min(group)
#         end = max(group)
#         key = f"{round(start, 4)}–{round(end, 4)}"
#         result[key] = group
#         current_bin += 1

#     return result


# def group_by_dynamic_range(numbers):
#     numbers = sorted(numbers)
#     min_val, max_val = min(numbers), max(numbers)

#     if min_val == max_val:
#         return {f"{min_val}–{max_val}": numbers}, [(min_val, max_val)]

#     # Dynamic number of groups (e.g., sqrt of data size)
#     num_groups = max(1, int(np.sqrt(len(numbers))))
#     bins = np.linspace(min_val, max_val, num_groups + 1)

#     # Step 1: Create initial groups
#     raw_groups = [[] for _ in range(num_groups)]
#     for num in numbers:
#         for i in range(num_groups):
#             if bins[i] <= num < bins[i + 1] or (i == num_groups - 1 and num == max_val):
#                 raw_groups[i].append(num)
#                 break

#     # Step 2: Merge small groups
#     merged_groups = []
#     i = 0
#     while i < len(raw_groups):
#         group = raw_groups[i]
#         if len(group) < 5:
#             if i + 1 < len(raw_groups):
#                 raw_groups[i + 1] = group + raw_groups[i + 1]
#             elif i > 0:
#                 merged_groups[-1] += group
#             else:
#                 merged_groups.append(group)
#         else:
#             merged_groups.append(group)
#         i += 1

#     # Step 3: Build new bin labels and collect bin edges
#     result = {}
#     bin_edges = []

#     for group in merged_groups:
#         if not group:
#             continue
#         start = min(group)
#         end = max(group)
#         key = f"{round(start, 4)}–{round(end, 4)}"
#         result[key] = group
#         bin_edges.append((start, end))

#     return result, bin_edges

import numpy as np
from scipy.stats import norm

def group_by_dynamic_range(numbers):
    numbers = sorted(numbers)
    min_val, max_val = min(numbers), max(numbers)

    if min_val == max_val:
        return {f"{min_val}–{max_val}": numbers}, [(min_val, max_val)], {f"{min_val}–{max_val}": 1.0}

    # Stats for CDF
    mean = np.mean(numbers)
    std = np.std(numbers)

    # Dynamic number of groups (e.g., sqrt of data size)
    num_groups = max(1, int(np.sqrt(len(numbers))))
    bins = np.linspace(min_val, max_val, num_groups + 1)

    # Step 1: Create initial groups
    raw_groups = [[] for _ in range(num_groups)]
    for num in numbers:
        for i in range(num_groups):
            if bins[i] <= num < bins[i + 1] or (i == num_groups - 1 and num == max_val):
                raw_groups[i].append(num)
                break

    # Step 2: Merge small groups
    merged_groups = []
    i = 0
    while i < len(raw_groups):
        group = raw_groups[i]
        if len(group) < 5:
            if i + 1 < len(raw_groups):
                raw_groups[i + 1] = group + raw_groups[i + 1]
            elif i > 0:
                merged_groups[-1] += group
            else:
                merged_groups.append(group)
        else:
            merged_groups.append(group)
        i += 1

    # Step 3: Build new bin labels, edges, and CDF
    result = {}
    bin_edges = []
    cdf_by_group = {}

    for group in merged_groups:
        if not group:
            continue
        start = min(group)
        end = max(group)
        key = f"{round(start, 4)}–{round(end, 4)}"
        result[key] = group
        bin_edges.append((start, end))
        
        # Compute CDF portion for this bin
        cdf_value = norm.cdf(end, loc=mean, scale=std) - norm.cdf(start, loc=mean, scale=std)
        cdf_by_group[key] = cdf_value

    return result, bin_edges, cdf_by_group

def perform_shapiro_wilk_normality_test(residuals, file_handle):
    """
    Performs the Shapiro-Wilk test for normality on a set of residuals.

    Args:
        residuals (np.ndarray): A 1-D array of regression residuals.

    Returns:
        tuple: A tuple containing the Shapiro-Wilk test statistic and the p-value.
               Returns (None, None) if the input has fewer than 3 observations.
    """
    if len(residuals) < 3:
        print("Warning: Shapiro-Wilk test requires at least three observations.")
        file_handle.write("Warning: Shapiro-Wilk test requires at least three observations.\n")
        return None, None

    # Calculate and log mean and std
    mean_val = np.mean(residuals)
    std_val = np.std(residuals)
    print(f"Mean: {mean_val:.4f}")
    print(f"Standard Deviation: {std_val:.4f}\n")

    file_handle.write(f"Mean: {mean_val:.4f}\n")
    file_handle.write(f"Standard Deviation: {std_val:.4f}\n")

    statistic, p_value = shapiro(residuals)

    print(f"Shapiro-Wilk Statistic: {statistic:.4f}")
    file_handle.write(f"Shapiro-Wilk Statistic: {statistic:.4f}\n")
    print(f"Shapiro-Wilk P-value: {p_value:.4f}")
    file_handle.write(f"Shapiro-Wilk P-value: {p_value:.4f}\n")

    if statistic is not None:
        alpha = 0.05
        print(f"\nSignificance Level (alpha): {alpha}")
        file_handle.write(f"\nSignificance Level (alpha): {alpha}\n")
        if p_value <= alpha:
            print("The Shapiro-Wilk test suggests that the residuals are likely NOT normally distributed (p <= alpha).")
            file_handle.write("The Shapiro-Wilk test suggests that the residuals are likely NOT normally distributed (p <= alpha).\n")
        else:
            print("The Shapiro-Wilk test suggests that we do not have strong evidence to reject the normality of the residuals (p > alpha).")
            file_handle.write("The Shapiro-Wilk test suggests that we do not have strong evidence to reject the normality of the residuals (p > alpha).\n")

def perform_kolmogorov_smirnov_normality_test(residuals, file_handle):

    # # Standardize the residuals
    # standardized_residuals = (residuals - np.mean(residuals)) / np.std(residuals)

    # # Perform K-S test against standard normal distribution
    # ks_stat, p_value = kstest(standardized_residuals, 'norm')
    # # Test if it comes from a standard normal distribution
    # ks_stat, p_value = kstest(residuals, 'norm')

    mu, sigma = np.mean(residuals), np.std(residuals)
    ks_stat, p_value = kstest(residuals, lambda x: norm.cdf(x, loc=mu, scale=sigma))

    print("K-S Statistic:", ks_stat)
    print("p-value:", p_value)

    if ks_stat is not None:
        alpha = 0.05
        print(f"\nSignificance Level (alpha): {alpha}")
        file_handle.write(f"\nSignificance Level (alpha): {alpha}\n")
        if p_value <= alpha:
            print("The Kolmogorov Smirnov test suggests that the residuals are likely NOT normally distributed (p <= alpha).")
            file_handle.write("The Kolmogorov Smirnov test suggests that the residuals are likely NOT normally distributed (p <= alpha).\n")
        else:
            print("The Kolmogorov Smirnov test suggests that we do not have strong evidence to reject the normality of the residuals (p > alpha).")
            file_handle.write("The Kolmogorov Smirnov test suggests that we do not have strong evidence to reject the normality of the residuals (p > alpha).\n")

def perform_cramervonmises_normality_test(residuals, file_handle):
    """
    Performs the one-sample Cramér-von Mises test for goodness of fit
    against a normal distribution.

    Args:
        residuals (np.ndarray): A 1-D array of regression residuals (actual - expected).

    Returns:
        tuple: A tuple containing the Cramér-von Mises statistic and the p-value.
               Returns (None, None) if the input has fewer than 2 observations.
    """
    if len(residuals) < 2:
        print("Warning: Cramer-von Mises test requires at least two observations.")
        file_handle.write("Warning: Cramer-von Mises test requires at least two observations.\n")
        return None, None

    # Fit a normal distribution to the residuals to estimate mean and standard deviation
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)

    # Perform the Cramér-von Mises test against this fitted normal distribution
    # Note: When parameters are estimated from the data, the p-value from
    # cramervonmises is generally not reliable.

    result = cramervonmises(residuals, 'norm', args=(mean_residual, std_residual))
    statistic = result.statistic
    pvalue = result.pvalue

    print(f"Cramér-von Mises Statistic: {statistic:.4f}")
    file_handle.write(f"Cramér-von Mises Statistic: {statistic:.4f}\n")
    print(f"P-value (Note: May not be reliable with estimated parameters): {pvalue:.4f}")
    file_handle.write(f"P-value (Note: May not be reliable with estimated parameters): {pvalue:.4f}\n")

    if statistic is not None:
        alpha = 0.05
        print(f"\nSignificance Level (alpha): {alpha}")
        file_handle.write(f"\nSignificance Level (alpha): {alpha}\n")
        if pvalue is not None and pvalue <= alpha:
            print("The p-value is less than or equal to alpha, suggesting the residuals may NOT be normally distributed.")
            file_handle.write("The p-value is less than or equal to alpha, suggesting the residuals may NOT be normally distributed.\n")
        else:
            print("The p-value is greater than alpha, suggesting we do not have strong evidence to reject the normality of the residuals.")
            file_handle.write("The p-value is greater than alpha, suggesting we do not have strong evidence to reject the normality of the residuals.\n")






