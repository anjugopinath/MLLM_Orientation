from plot_patch_analysis_llavaOV_qwen import main

category = "natural_cropped" # "natural_cropped" | "in_place_rotated"
if(category=="natural_cropped"):
    blended_shapes_list  =  ["dog","lizard","train","fish","indoor","beach"]
elif(category=="in_place_rotated"):
    blended_shapes_list = ["koala-beach","vase-indoor","vase-toaster-indoor"]

other_nums = [15, 30, 78, 118, 148, 173]
anchor = 9

origScales = ["scale1"]
scaleTitles = [""]
filename = "vision_tower_Degrees_labels_and_predictions.csv"
subdir1 = "visionEncAll"
subdir2 = "train_and_test_visionEnc"
make_compact = True # The control flag

modelnames_list = [["Qwen2.5-VL-7B-Instruct"]] #"llava-ov-qwen2-7b", "Qwen2.5-VL-7B-Instruct"
# blended_shapes_list  = ["dog_on_beach", "lizard_on_fish", "train_on_indoor"]
patch_analysis_modes_list = [["byModelWeight"], ["byAbsDiff"], ["byRandomLocations"]]

for modelnames in modelnames_list:
    if(modelnames[0] == "Qwen2.5-VL-7B-Instruct"):
        backbone = "vit"
    elif(modelnames[0] == "llava-ov-qwen2-7b"):
        backbone = "siglip"
    for blended_shape in blended_shapes_list:
        for patch_analysis_modes in patch_analysis_modes_list:
            
            main(other_nums, anchor, modelnames, blended_shape, origScales, scaleTitles,
                 patch_analysis_modes, filename, category, subdir1, subdir2, backbone,make_compact)