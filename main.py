# Imports
import os
import sys
import torch
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from dataset import OrientationDataset
from model_ridge_regression import RidgeRegressionModel
from model_mlp import MLPModel
from test import test_model
from utils import *

def main(model_name, shape_, scaleBlendedDir, scaleBlendedDirImages, model_type,background_rot=False):
    #/<dirname>/<name>/orientation
    # model_name = "LLAVA1.6" #Options - DINO, LLAVA (llava1.5), LLAVA1.6, baseline
    backbone = 'clip_vitl336' #Resnet50 (DINO), clip_vitl336 (llava)
    alphas = 10**np.linspace(10,-2,100)*0.5
    is_scaled = True #Whether the embeddings are standardized in the model
    is_blurred = True
    
    if(shape_==None):
        shape = "bird"  # rectangle, hexagon, star, triangle, insect, bird, dog, rectangle-triangle, rectangle-triangle_visionEncAll, "blended"
    else:
        shape = shape_
        category = "blended"
        
    print("Shape : ", shape)
    qs_type = "what_angle" #["what_angle", "what_is_color"]
    subset = "_visionEncAll" #_llavaAll, _visionEncAll, ComplVisionEnc
    angle = '1_degrees_FIXED' # Options : '1_degrees', '1_degrees'
    input_angle_folder = "1_degree_FIXED" #1_degree, 1_degree_FIXED

                        # 5_degrees : images rotated by 5 degrees; total 71 images, we pick one
                        # 1_degrees : images rotated by 1 degree; total 360 images
                        #cycle, i.e., upto 120 degree rotations - so, 25 images
                        #similarly, for 2_degrees, dataset size = 61 images since there are
                        # totally 180 images when rotated by 2 degrees upto 120 degrees rotation
    cross_testing = False
    #Has Gaussian blurring been applied to the edges of the shapes?
    if(is_blurred):
        blur_str = 'blurred'
    else:
        blur_str = 'unblurred'
    print("Blurred ? ", blur_str)
    if(is_scaled):
        scale_str = 'Scaled'
    else:
        scale_str = 'Unscaled'
    print("Scaled ? ", scale_str)
    #Rectangle, Triangle
    if(shape=="rectangle"):
        base_img_dir = f'/<dirname>/<name>/g-t_embedding_classifier/data/filtered/synthetic/Canva_{blur_str}'
    #Rectangle-Triangle or rectangle-triangle_visionEncAll
    elif(shape=="rectangle-triangle"):
        base_img_dir = f'/<dirname>/<name>/g-t_embedding_classifier/data/filtered/synthetic/two_shapes_{blur_str}'
    #Imagenet_Insect, #Imagenet_Bird, Imagenet_Dog
    elif(shape=="bird" or shape=="insect" or shape=="dog"):
        base_img_dir = "/<dirname>/<name>/g-t_embedding_classifier/data/filtered/natural/imagenet"
    elif(category=="blended"):
        print("inside blended")
        base_img_dir = f"/<dirname>/<name>/g-t_embedding_classifier/data/filtered/composition/composed_rotated/"#1_degree/{shape}/{scaleBlendedDir}"
        print("base_img_dir : ", base_img_dir)
    # img_dir_5 = '/<dirname>/<name>/g-t_embedding_classifier/data/filtered/synthetic/images/shapes_white_equi'
    # img_dir_2 = '/<dirname>/<name>/g-t_embedding_classifier/data/filtered/synthetic/images_2_degrees/shapes_white_equi'

    if(shape=="triangle"):
        num_samples_5 = 24
        num_samples_2 = 60
        img_dir_5 = os.path.join(base_img_dir,'Triangle','5_degrees')
        img_dir_2 = os.path.join(base_img_dir,'Triangle','2_degrees')
    elif(shape=="rectangle"):
        num_samples_5 = 36
        num_samples_2 = 90
        num_samples_1= 180
        img_dir_5 = os.path.join(base_img_dir,'Rectangle','5_degrees')
        img_dir_2 = os.path.join(base_img_dir,'Rectangle','2_degrees')
        img_dir_1 = os.path.join(base_img_dir,'Rectangle','1_degree')
    elif(shape=="rectangle-triangle"):
        num_samples_05 = 720
        num_samples_1 = 360
        num_samples_2 = 180
        num_samples_5 = 72
        num_samples_10 = 36
        img_dir_05 = os.path.join(base_img_dir,'rectangle-triangle','0.5_degree')
        img_dir_1 = os.path.join(base_img_dir,'rectangle-triangle','1_degree')
        img_dir_2 = os.path.join(base_img_dir,'rectangle-triangle','2_degrees')
        img_dir_5 = os.path.join(base_img_dir,'rectangle-triangle','5_degrees')
        img_dir_10 = os.path.join(base_img_dir,'rectangle-triangle','10_degrees')
    elif(shape=="bird" or shape=="insect" or shape=="dog"):
        num_samples_05 = 720
        num_samples_1 = 360
        num_samples_2 = 180
        num_samples_5 = 72
        num_samples_10 = 36
        img_dir_05 = os.path.join(base_img_dir,shape,'0.5_degree')
        img_dir_1 = os.path.join(base_img_dir,shape, '1_degree')
    elif(category=="blended"):
        num_samples_1 = 360
        num_samples_5 = 72

        img_dir_1 = os.path.join(base_img_dir, input_angle_folder, shape, scaleBlendedDirImages)
        img_dir_5 = os.path.join(base_img_dir, "5_degrees", shape, scaleBlendedDirImages)

    # elif(shape=="dog"):
    #     num_samples_1 = 360
    #     img_dir_1 = os.path.join(base_img_dir,'dog_unblurred','1_degree')

    # if angle=='5_degrees':  
    #     num_samples = 24

    # elif angle=='2_degrees':
    #     num_samples = 60

    # Define model-specific variables
    if model_name == "DINO":
        # Data Paths
        # embedding_dir_5 = f'/<dirname>/<name>/affine_tranformation/objects/DINO/dino_llava_output/white_{shape}_equi/embeddings'
        # embedding_dir_2 = f'/<dirname>/<name>/affine_tranformation/objects/DINO/dino_llava_output/white_{shape}_equi_2_degrees/embeddings'
        # embedding_dir_5 = f'/<dirname>/<name>/affine_tranformation/objects/DINO/dino_llava_output/canva_{blur_str}_{shape}_5_degrees/embeddings'
        # embedding_dir_2 = f'/<dirname>/<name>/affine_tranformation/objects/DINO/dino_llava_output/canva_{blur_str}_{shape}_2_degrees/embeddings'
        embedding_dir_5 = f'/<dirname>/<name>/affine_tranformation/objects/DINO/dino_llava_output/two_shapes_{blur_str}/{shape}/5_degrees/embeddings'
        embedding_dir_2 = f'/<dirname>/<name>/affine_tranformation/objects/DINO/dino_llava_output/two_shapes_{blur_str}/{shape}/2_degrees/embeddings'
        required_file_name = "transformer.encoder.layers[0].norm2.npy"
        # flattened_size = 18669 * 256
    elif model_name == "llava-v1.6-vicuna-13b":
        if(category=="blended" and subset=="_visionEncAll"):
            embedding_dir_1 = f"/<dirname>/<name>/g-t_embedding_classifier/llava1.6/LLaVA/llava/eval/pca_images/{qs_type}/{input_angle_folder}/{shape}/{scaleBlendedDir}/embeddings"

    elif model_name == "llava-v1.5-13b":
        # Data Paths
        # embedding_dir_5 = f'/<dirname>/<name>/g-t_embedding_classifier/llava/LLaVA/llava/eval/pca_images/white_{shape}_equi/embeddings'
        # embedding_dir_2 = f'/<dirname>/<name>/g-t_embedding_classifier/llava/LLaVA/llava/eval/pca_images/white_{shape}_equi_2_degrees/embeddings'
        if(shape=="rectangle"):
            embedding_dir_5 = f'/<dirname>/<name>/g-t_embedding_classifier/llava/LLaVA/llava/eval/pca_images/canva_{blur_str}_{shape}_5_degrees/embeddings'
            embedding_dir_2 = f'/<dirname>/<name>/g-t_embedding_classifier/llava/LLaVA/llava/eval/pca_images/canva_{blur_str}_{shape}_2_degrees/embeddings'
            embedding_dir_1 = f'/<dirname>/<name>/g-t_embedding_classifier/llava/LLaVA/llava/eval/pca_images/canva_{blur_str}_{shape}_1_degree/embeddings'
        elif(shape=="rectangle-triangle" and subset=="_llavaAll"):
            embedding_dir_05 = "/<dirname>/<name>/g-t_embedding_classifier/llava/LLaVA/llava/eval/pca_images/rectangle-triangle_05_degrees/embeddings"
            embedding_dir_1 = "/<dirname>/<name>/g-t_embedding_classifier/llava/LLaVA/llava/eval/pca_images/rectangle-triangle_1_degrees/embeddings" 
            embedding_dir_2 = "/<dirname>/<name>/g-t_embedding_classifier/llava/LLaVA/llava/eval/pca_images/rectangle-triangle_2_degrees/embeddings"
            embedding_dir_5 = "/<dirname>/<name>/g-t_embedding_classifier/llava/LLaVA/llava/eval/pca_images/rectangle-triangle_5_degrees/embeddings"
            embedding_dir_10 = "/<dirname>/<name>/g-t_embedding_classifier/llava/LLaVA/llava/eval/pca_images/rectangle-triangle_10_degrees/embeddings"
        elif(shape=="rectangle-triangle" and subset=="_visionEncAll"):
            embedding_dir_1 = f"/<dirname>/<name>/g-t_embedding_classifier/llava/LLaVA/llava/eval/pca_images/{qs_type}/rectangle-triangle_visionEncAll_1_degree/embeddings"
            embedding_dir_05 = f"/<dirname>/<name>/g-t_embedding_classifier/llava/LLaVA/llava/eval/pca_images/{qs_type}/rectangle-triangle_visionEncAll_05_degree/embeddings"
        elif(shape=="insect" and subset=="_visionEncAll"):
            embedding_dir_1 = f"/<dirname>/<name>/g-t_embedding_classifier/llava/LLaVA/llava/eval/pca_images/{qs_type}/imagenet_insect_visionEncAll_1_degree/embeddings"
            embedding_dir_05 = f"/<dirname>/<name>/g-t_embedding_classifier/llava/LLaVA/llava/eval/pca_images/{qs_type}/imagenet_insect_visionEncAll_05_degree/embeddings"
        elif(shape=="bird"):
            if(subset=="_visionEncAll"):
                embedding_dir_1 = f"/<dirname>/<name>/g-t_embedding_classifier/llava/LLaVA/llava/eval/pca_images/{qs_type}/imagenet_bird_visionEncAll_1_degree/embeddings"
            elif(subset=="ComplVisionEnc"):
                embedding_dir_1 = f"/<dirname>/<name>/g-t_embedding_classifier/llava/LLaVA/llava/eval/pca_images/what_angle/bird_ComplVisionEnc_1_degree/embeddings"
        elif(shape=="dog" and subset=="_visionEncAll"):
            embedding_dir_1 = f"/<dirname>/<name>/g-t_embedding_classifier/llava/LLaVA/llava/eval/pca_images/{qs_type}/imagenet_dog_visionEncAll_1_degree/embeddings"
        elif(category=="blended" and subset=="_visionEncAll"):
            embedding_dir_1 = f"/<dirname>/<name>/g-t_embedding_classifier/llava/LLaVA/llava/eval/pca_images/{qs_type}/{input_angle_folder}/{shape}/{scaleBlendedDir}/embeddings"
            embedding_dir_5 = f"/<dirname>/<name>/g-t_embedding_classifier/llava/LLaVA/llava/eval/pca_images/{qs_type}/5_degrees/{shape}/{scaleBlendedDir}/embeddings"
        # elif(shape=="dog"):
        #     embedding_dir_1 = "/<dirname>/<name>/g-t_embedding_classifier/llava/LLaVA/llava/eval/pca_images/dog_1_degree/embeddings" 
        # required_file_name = "encoder0_layernorm2.npy"
        # flattened_size = 577 * 1024
    elif model_name == "baseline":
        if(category=="blended" and subset=="_visionEncAll"):
            embedding_dir_5 = f"/<dirname>/<name>/g-t_embedding_classifier/llava/LLaVA/llava/eval/pca_images/{qs_type}/baseline/5_degrees/{shape}/{scaleBlendedDir}/embeddings"
            embedding_dir_1 = f"/<dirname>/<name>/g-t_embedding_classifier/llava/LLaVA/llava/eval/pca_images/{qs_type}/baseline/1_degree/{shape}/{scaleBlendedDir}/embeddings"

    # Data Transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image to 224x224
            transforms.ToTensor(),  # Convert the image to a tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize using ImageNet stats
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image to 224x224
            transforms.ToTensor(),  # Convert the image to a tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize using ImageNet stats
        ]),
    }

    if angle =='05_degrees':
        num_samples = num_samples_05
        img_dir = img_dir_05
        embedding_dir = embedding_dir_05
    elif angle =='1_degrees':
        num_samples = num_samples_1
        img_dir = img_dir_1
        embedding_dir = embedding_dir_1
    elif angle =='1_degrees_FIXED':
        num_samples = num_samples_1
        img_dir = img_dir_1
        embedding_dir = embedding_dir_1
    elif angle =='2_degrees':
        num_samples = num_samples_2
        img_dir = img_dir_2
        embedding_dir = embedding_dir_2
    elif angle =='5_degrees':
        num_samples = num_samples_5
        img_dir = img_dir_5
        embedding_dir = embedding_dir_5
    elif angle =='10_degrees':
        num_samples = num_samples_10
        img_dir = img_dir_10
        embedding_dir = embedding_dir_10

    test_size = str(int(.2*num_samples)) + '_test_samples'

    if(shape_==None):
        output_dir = f'output/{model_name}/{shape}/{subset}/{backbone}/{qs_type}/{angle}/{test_size}/{scale_str}/{model_type}/train_and_test'
        plot_dir = f'output/plots/{model_name}/{shape}/{subset}/{backbone}/{qs_type}/{angle}/{test_size}/{scale_str}/{model_type}'
    else:
        output_dir = f'output/{model_name}/{category}/{shape}/{angle}/{scaleBlendedDir}/{subset}/{backbone}/{qs_type}/{test_size}/{scale_str}/{model_type}/train_and_test'
        plot_dir = f'output/plots/{model_name}/{category}/{shape}/{angle}/{scaleBlendedDir}/{subset}/{backbone}/{qs_type}/{test_size}/{scale_str}/{model_type}'

    # Create directories if they do not exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

   
    print("embedding_dir : ",embedding_dir)
    print("num samples : ", num_samples)
    for file_name in os.listdir(embedding_dir):
        print("file_name : ", file_name)
        layername = file_name.split(".npy")[0]
        if file_name.endswith(".npy"):# and file_name == required_file_name:
        # if file_name.endswith("vision_model.encoder.layers.23.layer_norm2.npy") or file_name.endswith("vision_model.post_layernorm.npy"):
        #if file_name.endswith("lm_head.npy"):
            print("File Name: ", file_name)
            embedding_file_path = os.path.join(embedding_dir, file_name)
            print("embedding_file_path : ", embedding_file_path)
            # Load the .npy file
            data = np.load(embedding_file_path, allow_pickle=True)

            labels = extract_sorted_dict_keys(data, num_samples)
            print("len labels : ", len(labels))
            orig_labels = labels
            # print(data)

            print("img dir : ",img_dir)
            print("labels : ",labels)
            # Create Dataset
            dataset = OrientationDataset(labels=labels, img_dir=img_dir, shape=shape,
                                        embedding_file=embedding_file_path, output_dir = output_dir, transform=data_transforms['train'], background_rot=background_rot)

            print("len dataset : ", len(dataset))
            if not cross_testing:
                # Split Dataset
                train_indices, test_indices = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
                print(f"Length of train_indices: {len(train_indices)}")
                print(f"Length of test_indices: {len(test_indices)}")

            else:
                train_indices, __indices = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
                print(f"Length of train_indices: {len(train_indices)}")

            train_dataset = Subset(dataset, train_indices)
            
            # Save train and validation datasets
            save_dataset_to_csv(train_dataset, f'{output_dir}/{layername}_train_dataset.csv', "train")

            if(model_type == 'RidgeRegression'):
                # DataLoaders
                train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
            # elif(model_type == 'MLP'):
            #     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

            if(model_type == 'RidgeRegression'):
                # Define Ridge Regression Model
                model = RidgeRegressionModel(alphas, is_scaled)
            # elif(model_type == 'MLP'):
                # model = MLPModel(alphas, is_scaled)
                # model = MLPModel(input_dim=train_embeddings.shape[1], is_scaled=True)
                # model.fit(train_embeddings, train_labels)

            if(model_type=='RidgeRegression'):
                # Prepare data for Ridge Regression
                train_embeddings, train_labels = [], []

                # Extract embeddings and labels for Ridge Regression
                for inputs, embeddings, labels in train_loader:
                    train_embeddings.append(embeddings.view(-1).numpy())
                    train_labels.append(labels.numpy())


                # # Convert to numpy arrays
                # for i, emb in enumerate(train_embeddings):
                #     print(f"Index {i}: Type: {type(emb)}, Shape: {np.shape(emb)}")

                train_embeddings = np.array(train_embeddings)
                train_labels = np.array(train_labels).flatten()

            if(model_type=='RidgeRegression'):
                # Train the Ridge Regression model
                model_sin_alpha, model_cos_alpha, model_sin_coef, model_cos_coef = model.fit(train_embeddings, train_labels)

                #Save weights:
                model_save_dir = os.path.join(output_dir, "saved_models")
                os.makedirs(model_save_dir, exist_ok=True)
            # elif(model_type=='MLP'):
              
            #     X_train, X_val, y_train, y_val = train_test_split(
            #         train_embeddings, train_labels, test_size=0.1, random_state=42
            #     )
            #     model = MLPModel(input_dim=train_embeddings.shape[1], epochs=500)
            #     model.fit(X_train, y_train, X_val=X_val, y_val=y_val, patience=25)

            #     # model_sin_alpha, model_cos_alpha, model_sin_coef, model_cos_coef = model.fit(train_embeddings, train_labels)

            #     #Save weights:
            #     model_save_dir = os.path.join(output_dir, "saved_models")
            #     os.makedirs(model_save_dir, exist_ok=True)
            elif(model_type=='MLP'):
                # Extract embeddings and labels from the dataset directly
                embeddings = []
                labels = []
                for _, emb, lbl in train_dataset:
                    embeddings.append(emb.view(-1).numpy())
                    # labels.append(lbl.numpy())
                    if isinstance(labels, torch.Tensor):
                        labels.append(labels.numpy())
                    else:
                        labels.append(np.array(labels))
                embeddings = np.array(embeddings)
                labels = np.array(labels).flatten()

                # Split train/validation
                X_train, X_val, y_train, y_val = train_test_split(
                    embeddings, labels, test_size=0.1, random_state=42
                )

                # Initialize and train MLP on GPU
                # model = MLPModel(input_dim=embeddings.shape[1], epochs=500)
                model = MLPModel(
                    input_dim=train_embeddings.shape[1],
                    lr=1e-3,
                    epochs=200,
                    save_dir=output_dir  # this will save mlp_training_curve.csv
                )
                model.fit(X_train, y_train, X_val=X_val, y_val=y_val, patience=25)

                # Save model
                model_save_dir = os.path.join(output_dir, "saved_models")
                os.makedirs(model_save_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(model_save_dir, f"{layername}_mlp_model.pth"))


            if model_type == 'RidgeRegression':
                # Save the Ridge Regression model object using pickle
                with open(os.path.join(model_save_dir, f"{layername}_ridge_model.pkl"), "wb") as f:
                    pickle.dump(model, f)
            # elif model_type == 'MLP':
            #     # Save the PyTorch MLP model
            #     torch.save(model.state_dict(), os.path.join(model_save_dir, f"{layername}_mlp_model.pth"))
            if model_type == 'RidgeRegression':
                with open(f'{output_dir}/global_results.csv', mode="a", newline="") as file:
                    writer = csv.writer(file)
                    
                    # Append the metrics as new rows
                    writer.writerow([f" ***** {layername} ***** "])
                    writer.writerow([" -- Model Alpha Values -- "])
                    writer.writerow(["Sin"])
                    writer.writerow([model_sin_alpha])
                    # # Write Sin coefficients as individual rows
                    # for index, coef in model_sin_coef_series.items():
                    #     writer.writerow([index, coef])
                    writer.writerow(["Cos"])
                    writer.writerow([model_cos_alpha])
                    # # Write Sin coefficients as individual rows
                    # for index, coef in model_cos_coef_series.items():
                    #     writer.writerow([index, coef])
                with open(f'{output_dir}/sin_model_weights.csv', mode="a", newline="") as file:
                    writer = csv.writer(file)
                    # Append the metrics as new rows
                    writer.writerow([f" ***** {layername} ***** "])
                    writer.writerow([" -- Sin Model Coef (Weights) -- "])
                    # writer.writerow([model_sin_coef])
                    for i, w in enumerate(model_sin_coef):
                        writer.writerow([i, w])
                    plt.figure(figsize=(8, 5))
                    counts, bin_edges, _ = plt.hist(model_sin_coef, bins=20, edgecolor='black')
                    

                with open(f'{output_dir}/cos_model_weights.csv', mode="a", newline="") as file:
                    writer = csv.writer(file)
                    # Append the metrics as new rows
                    writer.writerow([f" ***** {layername} ***** "])
                    writer.writerow([" -- Cos Model Coef (Weights) -- "])
                    # writer.writerow([model_sin_coef])
                    for i, w in enumerate(model_cos_coef):
                        writer.writerow([i, w])
            

            # # Validate the model
            # val_predictions = model.predict(val_embeddings)

            # # Compute metrics
            # mae = mean_absolute_error(val_labels, val_predictions)
            # r2 = r2_score(val_labels, val_predictions)
            # print(f'Validation Mean Absolute Error: {mae:.2f} degrees')
            # print(f'Validation R^2 Score: {r2:.2f}')

            # Test the model
            if not cross_testing:
                test_dataset = Subset(dataset, test_indices)
                save_dataset_to_csv(test_dataset, f'{output_dir}/{layername}_test_dataset.csv', "test")
                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)
                test_model(model, test_loader, layername, output_dir, plot_dir)
            else:      
                if(angle == '2_degrees'):
                    img_dir = img_dir_5
                    embedding_file_path = os.path.join(embedding_dir_5, file_name)
                    num_samples = num_samples_5
                    # Load the .npy file
                    data = np.load(embedding_file_path, allow_pickle=True)
                    labels = extract_sorted_dict_keys(data, num_samples)
                    orig_labels = labels
                    # Filter out multiples of 10 (common multiples of 2 and 5)
                    orig_labels = [str(label) for label in orig_labels if int(label) % 10 != 0]

                    test_dataset = OrientationDataset(labels=orig_labels, img_dir=img_dir, shape=shape,
                                                    embedding_file=embedding_file_path, transform=data_transforms['val'], background_rot=background_rot)
                    
                    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
                    test_model(model, test_loader, layername, output_dir, plot_dir)

                elif(angle == '5_degrees'):
                    img_dir = img_dir_2
                    embedding_file_path = os.path.join(embedding_dir_2, file_name)
                    num_samples = num_samples_2
                    # Load the .npy file
                    data = np.load(embedding_file_path, allow_pickle=True)
                    labels = extract_sorted_dict_keys(data, num_samples)
                    orig_labels = labels
                    # Filter out multiples of 10 (common multiples of 2 and 5)
                    orig_labels = [str(label) for label in orig_labels if int(label) % 10 != 0]

                    test_dataset = OrientationDataset(labels=orig_labels, img_dir=img_dir, shape=shape,
                                                    embedding_file=embedding_file_path, transform=data_transforms['val'], background_rot=background_rot)
                    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
                    test_model(model, test_loader, layername, output_dir, plot_dir)

                save_dataset_to_csv(test_dataset, f'{output_dir}/{layername}_test_dataset.csv', "test")
