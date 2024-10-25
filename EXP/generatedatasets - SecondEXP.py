import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import skimage.draw

# Add LLMP module to system path
import sys
sys.path.append("../")  # Adds the current directory to the Python path

import LLMP as L

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import json
import uuid  # For generating unique IDs

# Define the main output directory
main_output_dir = "output"

# Subdirectories for images and JSON files
image_output_dir = os.path.join(main_output_dir, "images")
json_output_dir = os.path.join(main_output_dir, "json")

# Create directories if they don't exist
os.makedirs(image_output_dir, exist_ok=True)
os.makedirs(json_output_dir, exist_ok=True)

# List of tasks and their respective questions
tasks = {

    "data_to_barchart"
    "data_to_piechart"
    
}

# Number of images to generate for each task
num_images_per_task = 3500

# List to store all data from all tasks
combined_dataset = []

# Loop through each task
for task, question in tasks.items():
    print(f"Generating images and dataset for task: {task}")
    
    # Set up a loop to generate images and collect their labels
    for i in range(num_images_per_task):
        # Call the function to get the image and label
        image_array, label = L.GPImage.figure3(task)

        # Convert the boolean array to uint8 format (0 for False, 255 for True)
        image_array_uint8 = (image_array * 255).astype(np.uint8)

        # Convert the NumPy array to a PIL image
        pil_image = Image.fromarray(image_array_uint8)

        # Generate a unique ID for the image
        unique_id = str(uuid.uuid4())  # Generate a unique ID using UUID

        # Save the image with the unique ID
        image_filename = os.path.join(image_output_dir, f"{unique_id}.jpg")
        pil_image.save(image_filename)

        # Create a JSON entry for the dataset
        json_entry = {
            'id': unique_id,               # Unique ID for the image
            'image': f"{unique_id}.jpg",   # Image filename
            'question': question,          # Corresponding question for the task
            'value': label                 # The label generated for the image
        }

        # Append the JSON entry to the combined dataset list
        combined_dataset.append(json_entry)

# Save the combined dataset as a single JSON file in the JSON folder
combined_json_filename = "combined_dataset.json"
combined_json_filepath = os.path.join(json_output_dir, combined_json_filename)

with open(combined_json_filepath, 'w') as json_file:
    json.dump(combined_dataset, json_file, indent=4)

print(f"Images saved in '{image_output_dir}' and combined dataset saved as '{combined_json_filename}' in '{json_output_dir}'")

# Load the combined JSON dataset
json_file = "output/json/combined_dataset.json"
image_folder = "output/images"

with open(json_file, 'r') as f:
    data = json.load(f)

# Display the first 9 images and their corresponding info
for entry in data[:9]:
    image_path = os.path.join(image_folder, entry['image'])
    img = Image.open(image_path)
    img.show()  # This will open the image using the default image viewer

    print(f"ID: {entry['id']}")
    print(f"Question: {entry['question']}")
    print(f"Value: {entry['value']}")
    print(f"Image path: {image_path}")
    print("-" * 40)