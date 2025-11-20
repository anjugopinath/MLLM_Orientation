from perform_analysis import main

if __name__ == "__main__":

    shape = "blended"
    modelname = "llava-v1.6-vicuna-13b" #Options - llava-v1.5-13b, llava-v1.6-vicuna-13b
    # modelname = "llava-v1.6-vicuna-13b"
    if(shape== "blended"):
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
                    main(subshape, scaleBlendedDir, modelname)
    else:
        main(None, None)

