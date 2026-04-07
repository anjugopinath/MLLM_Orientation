from perform_analysis import main

if __name__ == "__main__":

    category = "in_place_rotated" #"blended" | "natural_cropped" | in_place_rotated
    if(category== "blended"):
        subset = "_visionEncAll"
    elif(category== "natural_cropped" or category== "in_place_rotated"):
        subset = "visionEncAll"
        subdir1 = "train_and_test_visionEnc"
    modelname = "Qwen2.5-VL-7B-Instruct" #Options - llava-v1.5-13b, llava-v1.6-vicuna-13b, llava-ov-qwen2-7b, Qwen2.5-VL-7B-Instruct
    # modelname = "llava-v1.6-vicuna-13b"
    if(category== "blended"):
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
                if("dog_on_beach" in subshape):
                    main(category, subset, subshape, scaleBlendedDir, modelname)
    elif(category== "natural_cropped"):
        shape_list = ['dog', 'train', 'lizard','indoor', 'beach', 'fish']
        for shape in shape_list:
            main(category, subset, shape, "scale1", modelname, subdir1)
    elif(category== "in_place_rotated"):
        shape_list = ['koala-beach','vase-indoor','vase-toaster-indoor']
        for shape in shape_list:
            main(category, subset, shape, "scale1", modelname, subdir1)
    else:
        main(None, None)

