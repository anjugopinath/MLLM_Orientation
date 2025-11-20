import json
import os
import numpy as np

blended_qs_list = [
        "By how much is the dog in the centre inside the circle rotated if it is known that the rotation is zero degrees when the dog's legs are vertical?",
        "By how much is the lizard in the centre inside the circle rotated if it is known that the rotation is zero degrees when the lizard is roughly horizontal with its tail pointing left and its head pointing right?",
        "By how much is the train in the centre inside the circle rotated if it is known that the rotation is zero degrees when the train is vertical with the train tracks at the bottom and the steam from the train going vertically upwards?",
        "By how much is the object in the centre inside the circle rotated if it is known that the rotation is zero degrees when the pointed tip of the triangle in the centre is pointing downwards?"
    ]

foreground_list = ['dog', 'lizard', 'train', 'rectangleTriangle']
background_list = ['fish', 'indoor', 'building', 'beach']

for foreground in foreground_list:
    for background in background_list:
        if("dog" in foreground):
            qs = blended_qs_list[0]
        elif("lizard" in foreground):
            qs = blended_qs_list[1]
        elif("train" in foreground):
            qs = blended_qs_list[2]
        elif("rectangleTriangle" in foreground):
            qs = blended_qs_list[3]
        
        shape = f"{foreground}_on_{background}"

        rotation = "360" #360 (rectangle-triangle, imagenet, blended), 120 (triangle), 180 (rectangle)
        increment = "5" #0.5, 1, 2, 5
        angles = list(range(0, int(rotation), int(increment)))
        output_filename = f"{shape}_blended_{rotation}_{increment}_degrees.jsonl"
        # original_extension = ".png"
        rotated_extension = ".png"
        qs_type = "what_angle"
        str_increment = "5_degrees" #1_degree, 5_degrees
        category = "blended"

        base_input_qs_path = f"/<dirname>/<name>/g-t_embedding_classifier/generate_files_for_llava/qs_list/{category}/input_qs/{qs_type}/{str_increment}/{shape}"
        os.makedirs(base_input_qs_path, exist_ok=True)
        question_id = 0
        angles[0] = 0
        # print("angles are : ", angles)
        # Initialize the list for JSON content
        json_lines = [] 
        for angle in angles:
            json_lines.append({
                "question_id": question_id,
                "image": f"{shape}_{angle}{rotated_extension}",
                "text": f"{qs}",
                "category": "conv"
            })
            question_id += 1

        # Convert each dictionary to a JSON string and write each as a single line
        # output_path = "/<dirname>/<name>/g-t_embedding_classifier/generate_files_for_llava/qs_list/bottle/input_qs"
        filename = os.path.join(base_input_qs_path,output_filename)
        # print("Category is : ", category)
        # print("Shape is : ", shape)
        # print("Rotation is : ", rotation)
        # print("Increment is : ", increment)
        # print("Question is : ", qs)
        # print(f"Writing to {filename}")
        with open(filename, "w") as json_file:
            for entry in json_lines:
                json_file.write(json.dumps(entry) + "\n")
