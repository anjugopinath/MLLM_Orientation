import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

def write_embedding_info(file_path, embedding_file, embedding):

    # Create the file if it does not exist
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            pass  # Just create the file
            
    with open(file_path, "r+") as f:
        content = f.read()
        # Convert embedding_file to string to avoid dtype mismatch
        embedding_file_str = str(embedding_file)  # Ensure it's a string
        
        # Check if the embedding file name is already in the file
        if f"{embedding_file_str}\n" not in content:
            f.write("embedding_file:\n")
            f.write(embedding_file_str + "\n")
            f.write(f"Before shape : {embedding.shape}\n")
            
            embedding = embedding.flatten()  # Ensure it's a 1D array
            f.write(f"After shape : {embedding.shape}\n")


# Custom Dataset Class
class OrientationDataset(Dataset):
    def __init__(self, labels, shape, embedding_file, output_dir, transform=None, background_rot=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            embedding_dir (string): Directory with precomputed embeddings as .npy files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        
        self.labels = labels
        # self.img_dir = img_dir
        if background_rot:
            self.shape = shape.replace("_b.5Rot_f.5Rot_1degreeTotal", "").replace("_bRot_fStat", "")
        else:
            self.shape = shape
        self.embedding_file = embedding_file
        self.output_dir = output_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        # Get the label (string) from the 'labels' list using the index
        label = self.labels[idx]
        # print("label inside dataset : ", label)

        if isinstance(label, torch.Tensor):
            label = str(label.item()).split(".")[0]
        
        # Construct the image file name based on the label
        if isinstance(label, torch.Tensor):
            # print(str(label.item()))
            image_index = self.shape + "_" + str(label.item()).split(".")[0] + ".png"
        else:
            image_index = self.shape + "_" + str(label) + ".png"
        # print("shape inside dataset : ", self.shape)
        # img_name = os.path.join(self.img_dir, image_index)
        
        # # Load the image
        # image = Image.open(img_name).convert('RGB')  # Ensure 3 channels
        
        
        # Load the embeddings dictionary from the .npy file
        embeddings_dict = np.load(self.embedding_file, allow_pickle=True).item()
        
        # Get the corresponding embedding using the label
        embedding = embeddings_dict.get(label, None)  # Get the embedding for the label
        file_path = f"{self.output_dir}/embedding_shape.txt"
        write_embedding_info(file_path, self.embedding_file, embedding)
        embedding = embedding.flatten()  # Ensure it's a 1D array
        
        
        if embedding is None:
            raise ValueError(f"Embedding for label {label} not found in the embeddings dictionary.")

        # Assuming that you want to predict the orientation label from the labels (instead of reading from another source)
        # You can modify the following if you have the label's orientation elsewhere
        label_value = float(self.labels[idx])  # If 'self.labels' contains orientation, use it directly

        # # Apply any transformations on the image
        # if self.transform:
        #     image = self.transform(image)
        # image = image.float()

        # return image, torch.tensor(embedding, dtype=torch.float32), label_value
        return torch.tensor(embedding, dtype=torch.float32), label_value
