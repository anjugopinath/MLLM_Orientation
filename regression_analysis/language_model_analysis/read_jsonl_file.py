# import sys
# import json

# # Path to your .jsonl file
# input_file = "1.5_dog_on_beach_scale1.jsonl" #1.5_dog_on_beach_scale1.jsonl, 1.6_dog_on_beach_scale1.jsonl

# # Question IDs you want to extract
# target_ids = {
#     3, 5, 9, 15, 22, 25, 30, 33, 39, 42, 45, 46, 55, 56, 57, 63, 72, 73, 76, 77, 78,
#     82, 84, 90, 93, 94, 108, 113, 116, 119, 124, 126, 139, 140, 147, 148, 157, 180,
#     197, 198, 199, 203, 209, 210, 211, 220, 222, 224, 232, 237, 248, 267, 272, 283,
#     284, 285, 286, 297, 300, 302, 305, 324, 325, 333, 334, 339, 342, 347, 351, 354,
#     356, 358
# }

# # Output list for the extracted texts
# extracted_texts = []

# # # Read and process each line in the JSONL file
# # with open(input_file, "r", encoding="utf-8") as f:
# #     for line in f:
# #         try:
# #             data = json.loads(line)
# #             print(data.get("question_id"))
# #         except json.JSONDecodeError:
# #             continue  # skip malformed lines

# # sys.exit(1)
# # Read and process each line in the JSONL file
# with open(input_file, "r", encoding="utf-8") as f:
#     for line in f:
#         try:
#             data = json.loads(line)
#             if data.get("question_id") in target_ids:
#                 extracted_texts.append(data.get("text", ""))
#         except json.JSONDecodeError:
#             continue  # skip malformed lines

# print(f"Extracted {len(extracted_texts)} texts matching target IDs.")
# # Option 1: print results
# for text in extracted_texts:
#     print(text, "\n---")

import json
import csv

# Path to your .jsonl file
input_file = "1.6_dog_on_beach_scale1.jsonl"  # or 1.6_dog_on_beach_scale1.jsonl
model_version = input_file.split("_")[0]  # Extract model version from filename
output_file = f"{model_version}_filtered_texts.csv"

# Question IDs you want to extract
target_ids = {
    3, 5, 9, 15, 22, 25, 30, 33, 39, 42, 45, 46, 55, 56, 57, 63, 72, 73, 76, 77, 78,
    82, 84, 90, 93, 94, 108, 113, 116, 119, 124, 126, 139, 140, 147, 148, 157, 180,
    197, 198, 199, 203, 209, 210, 211, 220, 222, 224, 232, 237, 248, 267, 272, 283,
    284, 285, 286, 297, 300, 302, 305, 324, 325, 333, 334, 339, 342, 347, 351, 354,
    356, 358
}

# List to store extracted rows (as dicts)
rows = []

# Read and process each line in the JSONL file
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        try:
            data = json.loads(line)
            qid = data.get("question_id")
            if qid in target_ids:
                text = data.get("text", "").replace("\n", " ").strip()
                rows.append({"question_id": qid, "text": text})
        except json.JSONDecodeError:
            continue  # skip malformed lines

# Write results to CSV
with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["question_id", "text"])
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ Extracted {len(rows)} matching entries written to '{output_file}'")

