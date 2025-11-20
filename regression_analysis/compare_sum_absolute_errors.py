import os

modelnames = ["llava-v1.5-13b", "llava-v1.6-vicuna-13b"]
blended_shapes = ["dog_on_beach", "lizard_on_fish", "train_on_indoor"]
scales = ["scale1", "scale4", "scale16"]

llava1_6_parent_path = "/<dirname>/<name>/affine_tranformation/objects/DINO/regressor/Linear_Regression/output/llava-v1.6-vicuna-13b/blended"
llava1_5_parent_path = "/<dirname>/<name>/affine_tranformation/objects/DINO/regressor/Linear_Regression/output/llava-v1.5-13b/blended"

for modelname in modelnames:
    for blended_shape in blended_shapes:
        for scale in scales:
            deg_pred_filename = f"{blended_shape}/1_degrees/{scale}/_visionEncAll/clip_vitl336/what_angle/72_test_samples/Scaled/RidgeRegression/train_and_test/vision_model.encoder.layers.23.layer_norm2_Degrees_labels_and_predictions.csv"
            
            if modelname == "llava-v1.6-vicuna-13b":
                final_file_path = os.path.join(llava1_6_parent_path, deg_pred_filename)
            elif modelname == "llava-v1.5-13b":
                final_file_path = os.path.join(llava1_5_parent_path, deg_pred_filename)
            
            # Read the file and extract sum of absolute errors
            if os.path.exists(final_file_path):
                with open(final_file_path, 'r') as f:
                    lines = f.readlines()
                
                sum_abs_error = None
                for line in lines:
                    if "Sum Absolute Error" in line:
                        sum_abs_error = float(line.split(":")[1].split()[0])
                        break
                
                print(f"Model: {modelname}, Shape: {blended_shape}, Scale: {scale}, Sum Absolute Error: {sum_abs_error}")
            else:
                print(f"File not found: {final_file_path}")

'''
Model: llava-v1.5-13b, Shape: dog_on_beach, Scale: scale1, Sum Absolute Error: 96.66
Model: llava-v1.5-13b, Shape: dog_on_beach, Scale: scale4, Sum Absolute Error: 68.39
Model: llava-v1.5-13b, Shape: dog_on_beach, Scale: scale16, Sum Absolute Error: 46.91
Model: llava-v1.5-13b, Shape: lizard_on_fish, Scale: scale1, Sum Absolute Error: 120.74
Model: llava-v1.5-13b, Shape: lizard_on_fish, Scale: scale4, Sum Absolute Error: 103.58
Model: llava-v1.5-13b, Shape: lizard_on_fish, Scale: scale16, Sum Absolute Error: 77.24
Model: llava-v1.5-13b, Shape: train_on_indoor, Scale: scale1, Sum Absolute Error: 93.18
Model: llava-v1.5-13b, Shape: train_on_indoor, Scale: scale4, Sum Absolute Error: 56.41
Model: llava-v1.5-13b, Shape: train_on_indoor, Scale: scale16, Sum Absolute Error: 39.6
Model: llava-v1.6-vicuna-13b, Shape: dog_on_beach, Scale: scale1, Sum Absolute Error: 48.3
Model: llava-v1.6-vicuna-13b, Shape: dog_on_beach, Scale: scale4, Sum Absolute Error: 44.0
Model: llava-v1.6-vicuna-13b, Shape: dog_on_beach, Scale: scale16, Sum Absolute Error: 23.32
Model: llava-v1.6-vicuna-13b, Shape: lizard_on_fish, Scale: scale1, Sum Absolute Error: 70.86
Model: llava-v1.6-vicuna-13b, Shape: lizard_on_fish, Scale: scale4, Sum Absolute Error: 47.69
Model: llava-v1.6-vicuna-13b, Shape: lizard_on_fish, Scale: scale16, Sum Absolute Error: 48.3
Model: llava-v1.6-vicuna-13b, Shape: train_on_indoor, Scale: scale1, Sum Absolute Error: 55.61
Model: llava-v1.6-vicuna-13b, Shape: train_on_indoor, Scale: scale4, Sum Absolute Error: 34.21
Model: llava-v1.6-vicuna-13b, Shape: train_on_indoor, Scale: scale16, Sum Absolute Error: 32.59
'''

