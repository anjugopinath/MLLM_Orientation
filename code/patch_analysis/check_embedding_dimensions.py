import os
import numpy as np

blended_shapes = ["dog_on_beach", "lizard_on_fish", "train_on_indoor"]
modelnames = ["llava-v1.5-13b", "llava-v1.6-vicuna-13b"]
llava1_6_parent_path = f"/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/llava1.6/LLaVA/llava/eval/pca_images/what_angle/1_degree"
llava1_5_parent_path = f"/s/red/a/nobackup/vision/anju/g-t_embedding_classifier/llava/LLaVA/llava/eval/pca_images/what_angle/1_degree"
embedding_filename = "scale1/embeddings/vision_model.encoder.layers.23.layer_norm2.npy"

for modelname in modelnames:
    for blended_shape in blended_shapes:
        print(f"Checking embedding dimensions for model {modelname} and blended shape {blended_shape}")
        if modelname == "llava-v1.6-vicuna-13b":
            embedding_path = os.path.join(llava1_6_parent_path, blended_shape, embedding_filename)
        elif modelname == "llava-v1.5-13b":
            embedding_path = os.path.join(llava1_5_parent_path, blended_shape, embedding_filename)        

        embedding = np.load(embedding_path, allow_pickle=True).item()
        embedding_0 = embedding['0']
        print(f"Embedding shape for {modelname} --- {blended_shape}: {embedding_0.shape}")

'''
LLaVA 1.5:
    Checking embedding dimensions for model llava-v1.5-13b and blended shape dog_on_beach
    Embedding shape for llava-v1.5-13b --- dog_on_beach: torch.Size([1, 576, 1024])
    Checking embedding dimensions for model llava-v1.5-13b and blended shape lizard_on_fish
    Embedding shape for llava-v1.5-13b --- lizard_on_fish: torch.Size([1, 576, 1024])
    Checking embedding dimensions for model llava-v1.5-13b and blended shape train_on_indoor
    Embedding shape for llava-v1.5-13b --- train_on_indoor: torch.Size([1, 576, 1024])
LLaVA 1.6:
    Checking embedding dimensions for model llava-v1.6-vicuna-13b and blended shape dog_on_beach
    Embedding shape for llava-v1.6-vicuna-13b --- dog_on_beach: torch.Size([5, 576, 1024])
    Checking embedding dimensions for model llava-v1.6-vicuna-13b and blended shape lizard_on_fish
    Embedding shape for llava-v1.6-vicuna-13b --- lizard_on_fish: torch.Size([3, 576, 1024])
    Checking embedding dimensions for model llava-v1.6-vicuna-13b and blended shape train_on_indoor
    Embedding shape for llava-v1.6-vicuna-13b --- train_on_indoor: torch.Size([3, 576, 1024])
'''