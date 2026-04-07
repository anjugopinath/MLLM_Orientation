# import pandas as pd
# import numpy as np

# # Path to your Excel file
# file_path = "1.5_filtered_texts_angle.xlsx" #1.5_filtered_texts_angle.xlsx, 1.6_filtered_texts_angle.xlsx
# model_version = file_path.split("_")[0]  # Extract model version from filename
# # Define threshold
# threshold = 5

# # Read the specific sheet
# df = pd.read_excel(file_path, sheet_name="angle")

# # Extract the 'angle' column
# angles = df["angle"]

# # Print or use it
# print(angles)

# # Count unique values and their frequencies
# value_counts = angles.value_counts(dropna=False)

# # Print results
# print("Number of unique values:", angles.nunique(dropna=False))
# print("\nValue counts:")
# print(value_counts)

# # Extract the relevant columns
# question_ids = df["question_id"]
# angles = df["angle"]

# # Convert to numeric (in case they’re strings), coercing non-numeric to NaN
# question_ids = pd.to_numeric(question_ids, errors="coerce")
# angles = pd.to_numeric(angles, errors="coerce")

# # Compute absolute difference
# diff = np.abs(question_ids - angles)

# # Count correct and incorrect
# correct = (diff <= threshold).sum()
# incorrect = (diff > threshold).sum()

# # Print results
# print(f"Number of correct answers (|diff| <= {threshold}): {correct}")
# print(f"Number of incorrect answers (|diff| > {threshold}): {incorrect}")

import pandas as pd
import numpy as np

# Path to your Excel file
file_path = "1.5_filtered_texts_angle.xlsx"  # e.g., 1.5_filtered_texts_angle.xlsx, 1.6_filtered_texts_angle.xlsx
model_version = file_path.split("_")[0]  # Extract model version from filename

# Define threshold
threshold = 45

# Read the specific sheet
df = pd.read_excel(file_path, sheet_name="angle")

# Extract the 'angle' column
angles = df["angle"]

# Print or use it
print(angles)

# Count unique values and their frequencies
value_counts = angles.value_counts(dropna=False)

# Print results
print("Number of unique values:", angles.nunique(dropna=False))
print("\nValue counts:")
print(value_counts)

# Extract the relevant columns
question_ids = df["question_id"]
angles = df["angle"]

# Convert to numeric (in case they’re strings), coercing non-numeric to NaN
question_ids = pd.to_numeric(question_ids, errors="coerce")
angles = pd.to_numeric(angles, errors="coerce")

# Compute absolute difference
diff = np.abs(question_ids - angles)

# Identify correct and incorrect indices
correct_mask = diff <= threshold
incorrect_mask = diff > threshold

# Count correct and incorrect
correct = correct_mask.sum()
incorrect = incorrect_mask.sum()

# Print results
print(f"\nNumber of correct answers (|diff| <= {threshold}): {correct}")
print(f"Number of incorrect answers (|diff| > {threshold}): {incorrect}")

# Print question_ids for correct answers
print("\nQuestion IDs for correct answers:")
print(df.loc[correct_mask, "question_id"].to_list())

print("NaN count:", diff.isna().sum())
print("Rows excluded due to NaN:", len(df) - (correct + incorrect))

