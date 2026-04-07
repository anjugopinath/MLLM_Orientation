import os
import csv
import numpy as np

dataset_category = "blended" # multiview_rotated | natural_cropped | in_place_rotated | grid_lines_180 | grid_lines_90 | blended
modelname = "llava-v1.6-vicuna-13b" # llava-ov-qwen2-7b | Qwen2.5-VL-7B-Instruct, llava-v1.5-13b | llava-v1.6-vicuna-13b   
# print("hello")
if(dataset_category=="in_place_rotated"):
    if(modelname=="llava-ov-qwen2-7b"):
        parent_path1 = "/s/red/a/nobackup/vision/anju/affine_tranformation/objects/DINO/regressor/Linear_Regression/output/llava-ov-qwen2-7b/in_place_rotated/"
    elif(modelname=="Qwen2.5-VL-7B-Instruct"):
        parent_path1 = "/s/red/a/nobackup/vision/anju/affine_tranformation/objects/DINO/regressor/Linear_Regression/output/Qwen2.5-VL-7B-Instruct/in_place_rotated/"
    dataset_list = ["koala-beach","vase-indoor","vase-toaster-indoor"]

elif(dataset_category=="natural_cropped"):
    if(modelname=="llava-ov-qwen2-7b"):
        parent_path1 = "/s/red/a/nobackup/vision/anju/affine_tranformation/objects/DINO/regressor/Linear_Regression/output/llava-ov-qwen2-7b/natural_cropped"
    else:
        parent_path1 = "/s/red/a/nobackup/vision/anju/affine_tranformation/objects/DINO/regressor/Linear_Regression/output/Qwen2.5-VL-7B-Instruct/natural_cropped"
    dataset_list = ["dog","lizard","train","fish","indoor","beach"]

elif(dataset_category=="multiview_rotated"):
    parent_path1 = "/s/red/a/nobackup/vision/anju/affine_tranformation/objects/DINO/regressor/Linear_Regression/output/llava-ov-qwen2-7b/multiview_rotated"
    dataset_list = ["avg_embeddings"]

elif(dataset_category=="blended"):
    if(modelname=="llava-v1.5-13b"):
        parent_path1 = "/s/red/a/nobackup/vision/anju/affine_tranformation/objects/DINO/regressor/Linear_Regression/output/llava-v1.5-13b/blended"
    elif(modelname=="llava-v1.6-vicuna-13b"):
        parent_path1 = "/s/red/a/nobackup/vision/anju/affine_tranformation/objects/DINO/regressor/Linear_Regression/output/llava-v1.6-vicuna-13b/blended"
    dataset_list = ["dog_on_beach", "lizard_on_fish" , "train_on_indoor"]

elif(dataset_category=="grid_lines_180"):
    parent_path1 = "/s/red/a/nobackup/vision/anju/affine_tranformation/objects/DINO/regressor/Linear_Regression/output/llava-ov-qwen2-7b/grid_lines_180"
    # dataset_list = ["horizontal","vertical","dog_horizontal","dog_vertical"]
    parent_dataset_list = ["dog","train","lizard","beach","indoor","fish","white"]

    subdataset_list = [
        "horizontal",
        "vertical",
        "grid",
        "checkerboard",
        "checkerboard_dense",
        "counter_clockwise_grid",
        "curly_grid",
        "static_dense_checkerboard"
    ]

    dataset_list = [
        f"{parent}-{sub}"
        for parent in parent_dataset_list
        for sub in subdataset_list
        if not (
            parent == "white" and 
            sub in ["counter_clockwise_grid", "static_dense_checkerboard"]
        )
    ]

elif(dataset_category=="grid_lines_90"):
    print("here")
    parent_path1 = "/s/red/a/nobackup/vision/anju/affine_tranformation/objects/DINO/regressor/Linear_Regression/output/llava-ov-qwen2-7b/grid_lines_90"
    dataset_list = ["horizontal","vertical","dog_horizontal","dog_vertical"]

if(dataset_category=="grid_lines_90"):
    print("here1")
    parent_path2 = "1_degrees_FIXED/scale1/visionEncAll/siglip/what_angle/18_test_samples/Scaled/RidgeRegression/train_and_test_visionEnc"
    print("parent_path2 : ",parent_path2)
elif(dataset_category=="grid_lines_180"):
    parent_path2 = "1_degrees_FIXED/scale1/visionEncAll/siglip/what_angle/36_test_samples/Scaled/RidgeRegression/train_and_test_visionEnc"
else:
    if(modelname=="llava-ov-qwen2-7b"):
        parent_path2 = "1_degrees_FIXED/scale1/whole_model/siglip/what_angle/36_test_samples/Scaled/RidgeRegression/train_and_test_wholeModel"
    elif(modelname=="Qwen2.5-VL-7B-Instruct"):
        parent_path2 = "1_degrees_FIXED/scale1/visionEncAll/vit/what_angle/36_test_samples/Scaled/RidgeRegression/train_and_test_visionEnc"
    
if(modelname in ["llava-v1.5-13b","llava-v1.6-vicuna-13b"]):
    layer_list = ["vision_model.encoder.layers.23.layer_norm2"]
else:
    layer_list = ["vision_tower"]#["vision_tower","after_projector","input_embeds","llm_hidden"]

if(modelname in ["llava-v1.5-13b","llava-v1.6-vicuna-13b"]):
    scales = ["scale1", "scale4", "scale16"]
    for dataset in dataset_list:
        for layer in layer_list:
            for scale in scales:
                if(modelname=="llava-v1.5-13b"):
                    parent_path2 = f"1_degrees_FIXED/{scale}/_visionEncAll/clip_vitl336/what_angle/72_test_samples/Scaled/RidgeRegression/train_and_test"
                elif(modelname=="llava-v1.6-vicuna-13b"):
                    parent_path2 = f"1_degrees_FIXED/{scale}/_visionEncAll/clip_vitl336/what_angle/72_test_samples/Scaled/RidgeRegression/train_and_test"

                file_path = os.path.join(parent_path1, dataset, parent_path2, f"{layer}_Degrees_labels_and_predictions.csv")
                
                degrees_diff = []

                with open(file_path, "r") as f:
                    reader = csv.reader(f)
                    header = next(reader)  # skip header
                    
                    for row in reader:
                        # Stop when we reach the summary lines
                        if len(row) < 3:
                            break
                        try:
                            degrees_diff.append(abs(float(row[2])))
                        except ValueError:
                            break  # stop if summary text line is reached

                if degrees_diff:
                    mean_abs_error = np.mean(degrees_diff)
                    max_abs_error = np.max(degrees_diff)
                    min_abs_error = np.min(degrees_diff)

                    print(f"Dataset: {dataset}, Layer: {layer}, Scale: {scale}")
                    # print(f"Mean Absolute Error: {mean_abs_error:.4f}")
                    # print(f"Max Absolute Error: {max_abs_error:.4f}")
                    # print(f"Min Absolute Error: {min_abs_error:.4f}")
                    # print(f"Mean | Max | Min\n {mean_abs_error:.4f} {max_abs_error:.4f} {min_abs_error:.4f}")
                    # print(f"{'Mean':>6} {'Max':>6} {'Min':>6}")
                    # print(f"{mean_abs_error:6.4f} {max_abs_error:6.4f} {min_abs_error:6.4f}")
                    print("Mean")
                    print(f"{mean_abs_error:.4f}")
                    print("Max")
                    print(f"{max_abs_error:.4f}")
                    print("Min")
                    print(f"{min_abs_error:.4f}")
            
                else:
                    print(f"\nDataset: {dataset}, Layer: {layer}, Scale: {scale} - No valid data found.")
            print("---------------------------------------------")


else:
    for dataset in dataset_list:
        for layer in layer_list:
            file_path = os.path.join(parent_path1, dataset, parent_path2, f"{layer}_Degrees_labels_and_predictions.csv")
            
            degrees_diff = []

            with open(file_path, "r") as f:
                reader = csv.reader(f)
                header = next(reader)  # skip header
                
                for row in reader:
                    # Stop when we reach the summary lines
                    if len(row) < 3:
                        break
                    try:
                        degrees_diff.append(abs(float(row[2])))
                    except ValueError:
                        break  # stop if summary text line is reached

            if degrees_diff:
                mean_abs_error = np.mean(degrees_diff)
                max_abs_error = np.max(degrees_diff)
                min_abs_error = np.min(degrees_diff)

                print(f"Dataset: {dataset}, Layer: {layer}")
                # print(f"Mean Absolute Error: {mean_abs_error:.4f}")
                # print(f"Max Absolute Error: {max_abs_error:.4f}")
                # print(f"Min Absolute Error: {min_abs_error:.4f}")
                # print(f"Mean | Max | Min\n {mean_abs_error:.4f} {max_abs_error:.4f} {min_abs_error:.4f}")
                # print(f"{'Mean':>6} {'Max':>6} {'Min':>6}")
                # print(f"{mean_abs_error:6.4f} {max_abs_error:6.4f} {min_abs_error:6.4f}")
                print("Mean")
                print(f"{mean_abs_error:.4f}")
                print("Max")
                print(f"{max_abs_error:.4f}")
                print("Min")
                print(f"{min_abs_error:.4f}")
        
            else:
                print(f"\nDataset: {dataset}, Layer: {layer} - No valid data found.")
        print("---------------------------------------------")
