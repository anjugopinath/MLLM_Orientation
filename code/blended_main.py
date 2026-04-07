import sys
import copy
import time
from main import main

if __name__ == "__main__":

    category = "in_place_rotated" #blended, coco, navi, natural_cropped, natural_cropped_gaussian_blur, in_place_rotated, multiview_rotated, grid_lines_90, grid_lines_180
    synthetic_background = False
    background_rot = False
    if(synthetic_background):
        required_shape_list = ["dog"]#["dog","lizard","train"]
    elif(category== "coco" or category=="navi"):
        if(category== "coco"):
            required_shape_list = ["000000500634","000000532062"]
        if(category== "navi"):
            required_shape_list = ["fireengine"]
    elif(category=="natural_cropped"):
        required_shape_list = ["dog","lizard","train","fish","indoor","beach"]
    elif(category=="natural_cropped_gaussian_blur"):
        required_shape_list = ["dog","fish"]
    elif(category=="in_place_rotated"):
        # required_shape_list = ["000000500634","000000532062"]
        required_shape_list = ["koala-beach","vase-indoor","vase-toaster-indoor"]
    elif(category=="grid_lines_90" or category=="grid_lines_180"):
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

        required_shape_list = [
            f"{parent}-{sub}"
            for parent in parent_dataset_list
            for sub in subdataset_list
            if not (
                parent == "white" and 
                sub in ["counter_clockwise_grid", "static_dense_checkerboard"]
            )
        ]
    elif(category=="multiview_rotated"):
        required_shape_list = ["avg_embeddings"]
    else:
        required_shape_list = ["dog_on_beach"]#["dog_on_beach","lizard_on_fish","train_on_indoor"]
    model_type = 'RidgeRegression' #RidgeRegression, MLP
    subset = "visionEncAll" #visionEncAll, language_model, whole_model
    background_rot_type = "b.5Rot_f.5Rot_1degreeTotal" # "b.5Rot_f.5Rot_1degreeTotal", "bRot_fStat"
    # counter_rotation = True
    modelname = "Qwen2.5-VL-7B-Instruct" #Options - llava-v1.5-13b, llava-v1.6-vicuna-13b, llava-ov-qwen2-7b, Qwen2.5-VL-7B-Instruct
    perform_patch_analysis = True
    # other_nums = [2, 15, 39, 75, 111, 147, 183, 219, 255, 291, 327]
    # other_nums = [5, 15, 39, 72, 108, 147, 180, 220, 248, 286, 324]
    # other_nums = [5, 15, 72, 147, 220, 286, 324]
    # anchor = 3
    patch_analysis_mode = "byModelWeight" #["byModelWeight", "byAbsDiff", "byRandomLocations"]
    if(category== "blended"):
        foreground_list = ['dog', 'lizard', 'train', 'rectangleTriangle']
        if(synthetic_background):
            background_list = ['chessboard', 'grid_lines', 'horizontal_lines', 'vertical_lines']
            # background_list = ['grid_lines']
            if(background_rot):
                background_list = [bg + "_" + background_rot_type for bg in background_list]
        else:
            background_list = ['fish', 'indoor', 'building', 'beach']
            if(background_rot):
                background_list = [bg + "_" + background_rot_type for bg in background_list]
        subshape_list = []
        for foreground in foreground_list:
            for background in background_list:
                subshape_ = foreground + "_on_" + background
                subshape_list.append(subshape_)

        for subshape in subshape_list:
            for scale in range(60):
                # if scale not in [0]:#[0,1,3,7,15,31,59]:
                if scale not in [0]:
                # if scale not in [0,3,15]:
                # if scale not in [3,4,5]:
                    continue
                #Original
                # scaleBlendedDir = "scale" + str(scale+1)
                #Replaced embeddings in 15 from 3 based on highest absolute diff in embedding values at corresponding locations
                # scaleBlendedDir = "scale" + str(scale+1) + "_3into15_30countbins"
                #Replaced embeddings in 15 from 3 based on highest model weight/coef locations -> replace embedding values at corresponding locations
                if(perform_patch_analysis):
                    # if(subshape not in ["dog_on_beach", "lizard_on_fish", "train_on_indoor"]):
                    #     continue
                    if(modelname == "llava-v1.6-vicuna-13b"):
                        if(subshape == "dog_on_beach"):
                            num_bins_total = 2949120
                            step = [8000, 16000, 128000, 270000, 540000, 810000, 1080000, 1620000, 1890000, 2160000]
                        elif(subshape == "lizard_on_fish" or subshape == "train_on_indoor"):
                            num_bins_total = 1769472
                            step = [8000, 16000, 128000, 270000, 540000, 810000, 1080000, 1620000]
                    elif(modelname == "llava-v1.5-13b"):
                        num_bins_total = 589824
                        # step = 20000
                        # step = 60000
                        step = [15000, 30000, 60000, 120000, 180000, 240000, 300000, 360000, 420000, 480000]
                    # for num_bins in range(step, num_bins_total + 1, step):
                    print("subshape : ", subshape)
                    for num_bins in step:

                         for num in other_nums:
                            # pair_str = f"{anchor}into{num}_{num_bins}countbins"
                            pair_str = f"{anchor}into{num}"
                        
                            if(patch_analysis_mode== "byModelWeight"):
                                scaleBlendedDir = "scale" + str(scale+1) + f"_{pair_str}_{num_bins}modelweight" #100, 2000, 200000, 471860, 589824
                            elif(patch_analysis_mode== "byAbsDiff"):
                                scaleBlendedDir = "scale" + str(scale+1) + f"_{pair_str}_{num_bins}absdiff"
                            elif(patch_analysis_mode== "byRandomLocations"):
                                scaleBlendedDir = "scale" + str(scale+1) + f"_{pair_str}_{num_bins}random"

                            scaleBlendedDirImages = "scale" + str(scale+1)
                            # subshape="errored"
                            if("lizard_on_fish" in subshape):
                                print(f"Processing {subshape}, scale {scale+1}, num_bins {num_bins}")
                                start_time = time.time()  # start timer
                                main(modelname, subshape, scaleBlendedDir, scaleBlendedDirImages, model_type)
                                end_time = time.time()  # start timer
                                time_seconds = end_time - start_time
                                time_minutes = time_seconds / 60

                                print("Time taken:", time_seconds, "seconds")
                                print("Time taken:", time_minutes, "minutes")
                                print("--------------------------------------------------\n")
                            # sys.exit(1)
                else:
                    scaleBlendedDir = "scale" + str(scale+1)
                    scaleBlendedDirImages = "scale" + str(scale+1)
                    # if("dog_on_beach" in subshape):
                    # if("dog" in subshape):
                    # if(subshape in ["lizard","train"]):
                    # if(subshape in required_shape_list):
                    if any(subshape.startswith(prefix) for prefix in required_shape_list):
                        print(f"Processing {subshape}, scale {scale+1}")
                        start_time = time.time()  # start timer
                        main(modelname, subshape, scaleBlendedDir, scaleBlendedDirImages, model_type ,background_rot, subset)
                        end_time = time.time()  # start timer
                        time_seconds = end_time - start_time
                        time_minutes = time_seconds / 60

                        print("Time taken:", time_seconds, "seconds")
                        print("Time taken:", time_minutes, "minutes")
                        print("--------------------------------------------------\n")
    elif(category in ["navi", "coco", "natural_cropped", "natural_cropped_gaussian_blur", "in_place_rotated","multiview_rotated","grid_lines_90","grid_lines_180"]):
        subshape_list = copy.deepcopy(required_shape_list)
        for subshape in subshape_list:
            for scale in range(60):
                if scale not in [0]:
                    continue
                    
                if(perform_patch_analysis):
                    if(modelname == "llava-ov-qwen2-7b" or modelname == "Qwen2.5-VL-7B-Instruct"):
                        other_nums = [15, 30, 78, 118, 148, 173]
                        anchor = 9
                        if(modelname == "llava-ov-qwen2-7b"):
                            if(category=="natural_cropped" or category=="in_place_rotated"):
                                num_bins_total = 3359232
                                step = [8000, 16000, 128000, 270000, 540000, 810000, 1080000, 1620000, 1890000, 2160000, 3000000]
                        elif(modelname == "Qwen2.5-VL-7B-Instruct"):
                            if(category=="natural_cropped" or category=="in_place_rotated"):
                                if(subshape in ["fish","indoor","train"]):
                                    num_bins_total = 501760
                                    step = [2000, 5000, 8000, 16000, 25000, 75000, 128000, 270000, 335000, 500000]
                                else:
                                    num_bins_total = 829440
                                    step = [2000, 8000, 16000, 25000, 75000, 128000, 270000, 335000, 500000, 800000]
                        for num_bins in step:
                            for num in other_nums:
                                # pair_str = f"{anchor}into{num}_{num_bins}countbins"
                                pair_str = f"{anchor}into{num}"
                            
                                if(patch_analysis_mode== "byModelWeight"):
                                    scaleBlendedDir = "scale" + str(scale+1) + f"_{pair_str}_{num_bins}modelweight" #100, 2000, 200000, 471860, 589824
                                elif(patch_analysis_mode== "byAbsDiff"):
                                    scaleBlendedDir = "scale" + str(scale+1) + f"_{pair_str}_{num_bins}absdiff"
                                elif(patch_analysis_mode== "byRandomLocations"):
                                    scaleBlendedDir = "scale" + str(scale+1) + f"_{pair_str}_{num_bins}random"

                                scaleBlendedDirImages = "scale" + str(scale+1)
                                # subshape="errored"
                                # if("lizard_on_fish" in subshape):
                                print(f"Processing {subshape}, scale {scale+1}, num_bins {num_bins}")
                                start_time = time.time()  # start timer
                                main(modelname, subshape, scaleBlendedDir, scaleBlendedDirImages, model_type, category, background_rot, subset)
                                end_time = time.time()  # start timer
                                time_seconds = end_time - start_time
                                time_minutes = time_seconds / 60

                                print("Time taken:", time_seconds, "seconds")
                                print("Time taken:", time_minutes, "minutes")
                                print("--------------------------------------------------\n")
                else:
                    scaleBlendedDir = "scale" + str(scale+1)
                    scaleBlendedDirImages = "scale" + str(scale+1)
                    print(f"Processing {subshape}, scale {scale+1}")
                    start_time = time.time()  # start timer
                    main(modelname, subshape, scaleBlendedDir, scaleBlendedDirImages, model_type, category, background_rot, subset)
                    end_time = time.time()  # start timer
                    time_seconds = end_time - start_time
                    time_minutes = time_seconds / 60

                    print("Time taken:", time_seconds, "seconds")
                    print("Time taken:", time_minutes, "minutes")
                    print("--------------------------------------------------\n")


        else:
            for shape in required_shape_list:
                scaleBlendedDir = "scale1"
                scaleBlendedDirImages = "scale1"
                print(f"Processing {shape}")
                start_time = time.time()  # start timer
                main(modelname, shape, scaleBlendedDir, scaleBlendedDirImages, model_type, category, background_rot, subset)
                end_time = time.time()  # start timer
                time_seconds = end_time - start_time
                time_minutes = time_seconds / 60

                print("Time taken:", time_seconds, "seconds")
                print("Time taken:", time_minutes, "minutes")
                print("--------------------------------------------------\n")
    else:
        main(None, None)