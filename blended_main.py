import sys
import time
from main import main

if __name__ == "__main__":

    category = "blended"
    synthetic_background = True
    background_rot = False
    if(synthetic_background):
        required_shape_list = ["lizard","train"]#["dog","lizard","train"]
    else:
        required_shape_list = ["dog_on_beach","lizard_on_fish","train_on_indoor"]
    model_type = 'RidgeRegression' #RidgeRegression, MLP
    background_rot_type = "bRot_fStat" # "b.5Rot_f.5Rot_1degreeTotal", "bRot_fStat"
    # counter_rotation = True
    modelname = "llava-v1.5-13b" #Options - llava-v1.5-13b, llava-v1.6-vicuna-13b
    perform_patch_analysis = False
    # other_nums = [2, 15, 39, 75, 111, 147, 183, 219, 255, 291, 327]
    # other_nums = [5, 15, 39, 72, 108, 147, 180, 220, 248, 286, 324]
    other_nums = [5, 15, 72, 147, 220, 286, 324]
    anchor = 3
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
                    if(subshape not in ["dog_on_beach", "lizard_on_fish", "train_on_indoor"]):
                        continue
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
                        main(modelname, subshape, scaleBlendedDir, scaleBlendedDirImages, model_type ,background_rot)
                        end_time = time.time()  # start timer
                        time_seconds = end_time - start_time
                        time_minutes = time_seconds / 60

                        print("Time taken:", time_seconds, "seconds")
                        print("Time taken:", time_minutes, "minutes")
                        print("--------------------------------------------------\n")
    else:
        main(None, None)