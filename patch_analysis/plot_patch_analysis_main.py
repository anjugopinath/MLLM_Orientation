from plot_patch_analysis import main

anchor = 3
other_nums = [5, 15, 72, 147, 220, 286, 324]
origScales = ["scale1", "scale4", "scale16"]
scaleTitles = ["scale 1", "scale 2", "scale 3"]
filename = "vision_model.encoder.layers.23.layer_norm2_Degrees_labels_and_predictions.csv"
make_compact = True # The control flag

modelnames_list = [["llava-v1.6-vicuna-13b"]]#[["llava-v1.5-13b"],["llava-v1.6-vicuna-13b"]]
# blended_shapes_list  = ["dog_on_beach", "lizard_on_fish", "train_on_indoor"]
blended_shapes_list  = ["lizard_on_fish","train_on_indoor"]
patch_analysis_modes_list = [["byModelWeight"], ["byAbsDiff"], ["byRandomLocations"]]

for modelnames in modelnames_list:
    for blended_shape in blended_shapes_list:
        for patch_analysis_modes in patch_analysis_modes_list:
            
            
            main(other_nums, anchor, modelnames, blended_shape, origScales, scaleTitles,
                 patch_analysis_modes, filename, make_compact)

# modelnames = ["llava-v1.6-vicuna-13b"]  # ["llava-v1.5-13b", "llava-v1.6-vicuna-13b"]
# blended_shape = "lizard_on_fish"  # dog_on_beach, lizard_on_fish, train_on_indoor
# origScales = ["scale1", "scale4", "scale16"]
# scaleTitles = ["scale 1", "scale 2", "scale 3"]
# patch_analysis_modes = ["byRandomLocations"] # ["byModelWeight", "byAbsDiff", "byRandomLocations"]
# filename = "vision_model.encoder.layers.23.layer_norm2_Degrees_labels_and_predictions.csv"
# make_compact = True # The control flag
# main(other_nums, anchor, modelnames, blended_shape, origScales, scaleTitles,
#      patch_analysis_modes, filename, make_compact)