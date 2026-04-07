from stat_plots import main

if __name__ == "__main__":

    category = "natural_cropped" # natural_cropped, in_place_rotated
    # modelname = "llava-v1.5-13b" #Options - llava-v1.5-13b, llava-v1.6-vicuna-13b
    modelname = "Qwen2.5-VL-7B-Instruct" #Options - llava-v1.5-13b, llava-v1.6-vicuna-13b, llava-ov-qwen2-7b, Qwen2.5-VL-7B-Instruct
    if(modelname == "llava-v1.5-13b" or modelname == "llava-v1.6-vicuna-13b"):
        required_shape = "lizard_on_fish"
    if(category == "natural_cropped" or category == "in_place_rotated"):
        if(category == "natural_cropped"):
            shape_list = ['dog', 'train', 'lizard','indoor', 'beach', 'fish']
        elif(category == "in_place_rotated"):
            shape_list = ['koala-beach','vase-indoor','vase-toaster-indoor']
        subset = "visionEncAll"
        subdir1 = "train_and_test_visionEnc"
        scaleBlendedDir = "scale1"
    
    if(category== "blended"):
        subset = "_visionEncAll"
        foreground_list = ['dog', 'lizard', 'train', 'rectangleTriangle']
        background_list = ['fish', 'indoor', 'building', 'beach']
        subshape_list = []
        for foreground in foreground_list:
            for background in background_list:
                subshape_ = foreground + "_on_" + background
                subshape_list.append(subshape_)

        for subshape in subshape_list:
            for scale in range(60):
                if scale not in [0,3,15]:
                    continue
                scaleBlendedDir = "scale" + str(scale+1)
                if(required_shape in subshape):
                    print("Subshape:", subshape, "Scale:", scaleBlendedDir)
                    main(category,subshape, scaleBlendedDir, modelname, subset, required_shape, None)
                    print("--------------------------------------------------")

    elif(category== "natural_cropped" or category == "in_place_rotated"):
        for shape in shape_list:
            print("Shape:", shape)
            main(category, shape, scaleBlendedDir, modelname, subset, None, subdir1)
            print("--------------------------------------------------")
    else:
        main(None, None)

