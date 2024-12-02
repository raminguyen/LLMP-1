import os
import sys
import json
import uuid
import numpy as np
from PIL import Image

import os
import json

# Add custom module path
sys.path.append("../")
import LLMP as L

import os
import random
import uuid
import json
import numpy as np
from PIL import Image

import random
from PIL import Image
import matplotlib.pyplot as plt

import os
import random
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt


""" Generate EXP1 DATASET """

def generate_dataset_EXP1(main_output_dir="finetuning-EXP1-5000-test", images_per_task=5000):
    """
    Generate datasets for multiple tasks with no overlaps and save them to JSON files.

    Parameters:
        main_output_dir (str): Directory to save generated datasets and images.
        num_images_per_task (int): Number of images per task.

    Returns:
        None
    """
    # Define tasks and questions
    tasks = {
        "position_common_scale": "Please estimate the vertical position of the block relative to the line on the left (Top is 0, Bottom is 60). So the range is 0 - 60. No explanation.",
        "position_non_aligned_scale": "Please estimate the vertical position of the block relative to the line on the left (Top is 22, Bottom is 40). So the range is 22 - 40. No explanation.",
        "length": "Estimate the line length from top to bottom (range: 0-100). Number only. No explanation.",
        "direction": "Please estimate the direction of the line relative to the starting dot in the range 0 - 359 degrees. No explanation.",
        "angle": "Please estimate the angle (0-90 degrees). No explanation.",
        "area": "Estimate the area of a circle, ensuring your answer falls within the range of 3.14 to 5026.55 square units. Assume the circle fits within a 100x100 pixel image. Provide only the numeric value, no explanation.",
        "volume": "Estimate the volume of a cube, with your answer restricted to the range of 1 to 8000 cubic units. Assume the cube fits within a 100x100 pixel image. Provide only the numeric value, no explanation.",
        "curvature": "Please estimate the curvature of the line. (0 is no curvature - 1 is the maximum curvature) The more bend the line is, the higher the curvature. No explanation.",
        "shading": "Please estimate the shading density or texture density (range 0 to 100). No explanation."
    }

    # Define the target number of images for each task in the dataset
    train_target = images_per_task
    val_target = images_per_task // 5  # 20% of training size for validation
    test_target = images_per_task // 10  # 10% of training size for testing

    # Initialize dataset lists
    combined_dataset_training = []
    combined_dataset_validation = []
    combined_dataset_testing = []

    # Create directories if they don't exist
    image_output_dir = os.path.join(main_output_dir, "images")
    os.makedirs(image_output_dir, exist_ok=True)

    # Global label tracking for uniqueness across datasets
    train_labels = []
    val_labels = []
    test_labels = []

    for task, question in tasks.items():
        print(f"\n--- Generating images for task: {task} ---")

        # Initialize counters for each dataset
        train_counter = 0
        val_counter = 0
        test_counter = 0
        all_counter = 0

        while train_counter < train_target or val_counter < val_target or test_counter < test_target:
            all_counter += 1  # Track total iterations for debugging

            # Generate image and label using the custom module
            image_array, label = L.GPImage.figure1(task)

            # Determine which dataset the label belongs to
            pot = np.random.choice(3)
            if label in train_labels:
                pot = 0
            elif label in val_labels:
                pot = 1
            elif label in test_labels:
                pot = 2

            # Training dataset
            if pot == 0 and train_counter < train_target:
                if label not in train_labels:
                    train_labels.append(label)
                process_and_save_image(image_array, label, question, combined_dataset_training, image_output_dir, task)
                train_counter += 1

            # Validation dataset
            elif pot == 1 and val_counter < val_target:
                if label not in val_labels:
                    val_labels.append(label)
                process_and_save_image(image_array, label, question, combined_dataset_validation, image_output_dir, task)
                val_counter += 1

            # Test dataset
            elif pot == 2 and test_counter < test_target:
                if label not in test_labels:
                    test_labels.append(label)
                process_and_save_image(image_array, label, question, combined_dataset_testing, image_output_dir, task)
                test_counter += 1

        print(f"Task {task} generation completed with {all_counter} iterations.")

    # Save datasets to JSON files
    datasets = {
        "train": combined_dataset_training,
        "val": combined_dataset_validation,
        "test": combined_dataset_testing
    }
    save_datasets_to_json(datasets, main_output_dir)


def display_training_samples_exp1(main_output_dir="finetuning-EXP1-5000-test", num_images=2):
    """
    Display a specified number of random images for each task from the training dataset
    for Experiment 1 (position, length, direction, etc.).

    Parameters:
        main_output_dir (str): Directory where datasets and images are stored.
        num_images (int): Number of random images to display for each task.

    Returns:
        None
    """
    import os
    import json
    import random
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt

    # Define tasks and questions
    tasks = {
        "position_common_scale": "Please estimate the vertical position of the block relative to the line on the left (Top is 0, Bottom is 60). So the range is 0 - 60. No explanation.",
        "position_non_aligned_scale": "Please estimate the vertical position of the block relative to the line on the left (Top is 22, Bottom is 40). So the range is 22 - 40. No explanation.",
        "length": "Estimate the line length from top to bottom (range: 0-100). Number only. No explanation.",
        "direction": "Please estimate the direction of the line relative to the starting dot in the range 0 - 359 degrees. No explanation.",
        "angle": "Please estimate the angle (0-90 degrees). No explanation.",
        "area": "Estimate the area of a circle, ensuring your answer falls within the range of 3.14 to 5026.55 square units. Assume the circle fits within a 100x100 pixel image. Provide only the numeric value, no explanation.",
        "volume": "Estimate the volume of a cube, with your answer restricted to the range of 1 to 8000 cubic units. Assume the cube fits within a 100x100 pixel image. Provide only the numeric value, no explanation.",
        "curvature": "Please estimate the curvature of the line. (0 is no curvature - 1 is the maximum curvature) The more bend the line is, the higher the curvature. No explanation.",
        "shading": "Please estimate the shading density or texture density (range 0 to 100). No explanation."
    }

    # Load training dataset
    json_output_dir = os.path.join(main_output_dir, "json")
    train_file = os.path.join(json_output_dir, "train_dataset.json")
    
    with open(train_file, 'r') as f:
        train_dataset = json.load(f)

    print("\n" + "="*70)
    print("Displaying Training Dataset Samples")
    print("="*70)

    # Group images and labels by task
    task_images = {task: [] for task in tasks.keys()}
    task_labels = {task: [] for task in tasks.keys()}

    for entry in train_dataset:
        if entry['task'] in tasks:
            task_images[entry['task']].append(entry['image'])
            task_labels[entry['task']].append(entry['value'])


    # Display samples for each task
    for task, images in task_images.items():
        if len(images) < num_images:
            print(f"Not enough images for task '{task}' in training dataset.")
            continue

        print(f"\nTask: {task.upper()}")
        print(f"Displaying {num_images} random samples")
        print("-"*50)

        # Select random images and their corresponding labels
        indices = random.sample(range(len(images)), num_images)
        random_images = [images[i] for i in indices]
        random_labels = [task_labels[task][i] for i in indices]

        # Plot the images with enhanced styling
        fig, axes = plt.subplots(1, num_images, figsize=(15, 7))
        fig.suptitle(f"Training Dataset: {task.upper()}", fontsize=16, y=1.05)
        
        # Add background color
        fig.patch.set_facecolor('#f0f0f0')
        
        for i, (img_path, label) in enumerate(zip(random_images, random_labels)):
            img_path_full = os.path.join(main_output_dir, img_path)
            img = Image.open(img_path_full)
            
            if not isinstance(axes, np.ndarray):
                axes = [axes]
            
            axes[i].imshow(img, cmap="gray")
            axes[i].axis("off")
            axes[i].set_title(f"Sample {i+1}\nLabel: {label:.2f}", 
                            bbox=dict(facecolor='white', alpha=0.8),
                            pad=10)
            
            # Add border to each subplot
            for spine in axes[i].spines.values():
                spine.set_edgecolor('gray')
                spine.set_linewidth(2)

        # Display the corresponding prompt below the images
        prompt_text = tasks[task]
        plt.figtext(0.5, 0.02, prompt_text, 
                   wrap=True, 
                   horizontalalignment='center', 
                   fontsize=20,
                   bbox=dict(facecolor='white', 
                           edgecolor='gray',
                           alpha=0.8,
                           pad=10))
        
        plt.tight_layout()
        plt.show()
        print("\n")

""" Generate EXP2 DATASET """


def generate_dataset_EXP2(main_output_dir="finetuning-EXP2-5000-test", images_per_task=5000):
    """
    Generate balanced datasets for pie and bar tasks with noise added to images
    and ensure no overlap between training, validation, and test datasets.

    Parameters:
        main_output_dir (str): Directory to save generated datasets and images.
        images_per_task (int): Number of images to generate for each task in training, validation, and testing.

    Returns:
        None
    """
    import os
    import uuid
    import numpy as np
    from PIL import Image

    # Define the tasks and corresponding questions
    tasks = {
        "pie": (
            "The pie chart you are looking at is created as follows: "
            "First, create a list of five values where each value is between 3 and 39, and all values add up to 100. "
            "Next, divide each value in the list by the largest value, so that the largest value becomes 1.0. "
            "Now, look at the pie chart again. "
            "Identify the largest segment, which is marked with a dot. "
            "Go counterclockwise around the pie starting from the largest segment, estimating the ratio of the other four values to the maximum. "
            "Format your answer as [1.0, x.x, x.x, x.x, x.x]."
        ),
        "bar": (
            "The bar chart you are looking at is created as follows: "
            "First, create a list of five values where each value is between 3 and 39, and all values add up to 100. "
            "Next, divide each value in the list by the largest value, so that the largest value becomes 1.0. "
            "Now, look at the bar chart again. "
            "Identify the largest bar, which is marked with a dot. "
            "Move left to right along the bar chart starting from the largest bar, estimating the ratio of the other four values to the maximum. "
            "Format your answer as [1.0, x.x, x.x, x.x, x.x]."
        )
    }

    # Define the target number of images for each task in the dataset
    train_target = images_per_task
    val_target = images_per_task // 5  # 20% of training size for validation
    test_target = images_per_task // 10  # 10% of training size for testing

    # Initialize dataset lists
    combined_dataset_training = []
    combined_dataset_validation = []
    combined_dataset_testing = []

    # Create directories if they don't exist
    image_output_dir = os.path.join(main_output_dir, "images")
    os.makedirs(image_output_dir, exist_ok=True)

    # Global label tracking for uniqueness across datasets
    train_labels = []
    val_labels = []
    test_labels = []

    for task, question in tasks.items():
        print(f"\n--- Generating images for task: {task} ---")

        # Initialize counters for each dataset
        train_counter = 0
        val_counter = 0
        test_counter = 0
        all_counter = 0

        while train_counter < train_target or val_counter < val_target or test_counter < test_target:
            all_counter += 1  # Track total iterations for debugging

            # Generate image and label using the custom module
            image_array, label = L.GPImage.figure3(task)

            # Determine which dataset the label belongs to
            pot = np.random.choice(3)
            if label in train_labels:
                pot = 0
            elif label in val_labels:
                pot = 1
            elif label in test_labels:
                pot = 2

            # Training dataset
            if pot == 0 and train_counter < train_target:
                if label not in train_labels:
                    train_labels.append(label)
                process_and_save_image(image_array, label, question, combined_dataset_training, image_output_dir, task)
                train_counter += 1

            # Validation dataset
            elif pot == 1 and val_counter < val_target:
                if label not in val_labels:
                    val_labels.append(label)
                process_and_save_image(image_array, label, question, combined_dataset_validation, image_output_dir, task)
                val_counter += 1

            # Test dataset
            elif pot == 2 and test_counter < test_target:
                if label not in test_labels:
                    test_labels.append(label)
                process_and_save_image(image_array, label, question, combined_dataset_testing, image_output_dir, task)
                test_counter += 1

        print(f"Task {task} generation completed with {all_counter} iterations.")

    # Save datasets to JSON files
    datasets = {
        "train": combined_dataset_training,
        "val": combined_dataset_validation,
        "test": combined_dataset_testing
    }
    save_datasets_to_json(datasets, main_output_dir)

def display_training_samples_exp2(main_output_dir="finetuning-EXP2-5000-test", num_images=2):
    """
    Display a specified number of random images for each task and dataset 
    (training, validation, testing) after dataset generation, including rounded labels and prompts.

    Parameters:
        main_output_dir (str): Directory where datasets and images are stored.
        num_images (int): Number of random images to display for each task.

    Returns:
        None
    """
    import os
    import json
    import random
    from PIL import Image
    import matplotlib.pyplot as plt

   # Define tasks and prompts for pie and bar charts
    task_prompts = {
        "pie": (
            "The pie chart you are looking at is created as follows: "
            "First, create a list of five values where each value is between 3 and 39, and all values add up to 100. "
            "Next, divide each value in the list by the largest value, so that the largest value becomes 1.0. "
            "Now, look at the pie chart again. "
            "Identify the largest segment, which is marked with a dot. "
            "Go counterclockwise around the pie starting from the largest segment, estimating the ratio of the other four values to the maximum. "
            "Format your answer as [1.0, x.x, x.x, x.x, x.x]."
        ),
        "bar": (
            "The bar chart you are looking at is created as follows: "
            "First, create a list of five values where each value is between 3 and 39, and all values add up to 100. "
            "Next, divide each value in the list by the largest value, so that the largest value becomes 1.0. "
            "Now, look at the bar chart again. "
            "Identify the largest bar, which is marked with a dot. "
            "Move left to right along the bar chart starting from the largest bar, estimating the ratio of the other four values to the maximum. "
            "Format your answer as [1.0, x.x, x.x, x.x, x.x]."
        )
    }
    
    # Define dataset file paths
    json_output_dir = os.path.join(main_output_dir, "json")
    train_file = os.path.join(json_output_dir, "train_dataset.json")


    # Load datasets
    with open(train_file, 'r') as f:
        train_dataset = json.load(f)

    # Group datasets for display with visual indicators
    datasets = {
        "🔵 Training": train_dataset,

    }

    # Display random images for each dataset and task
    for dataset_name, dataset in datasets.items():
        print("\n" + "="*70)
        print(f"📸 Displaying random images for {dataset_name} dataset:")
        print("="*70)

        # Group images and labels by task
        task_images = {"pie": [], "bar": []}
        task_labels = {"pie": [], "bar": []}

        for entry in dataset:
            task_images[entry['task']].append(entry['image'])
            task_labels[entry['task']].append(entry['value'])

        # Display samples for each task
        for task, images in task_images.items():
            if len(images) < num_images:
                print(f"⚠️  Not enough images for task '{task}' in training dataset.")
                continue

            print(f"\n📌 Task: {task.upper()}")
            print(f"Displaying {num_images} random samples")
            print("-"*50)

            # Select random images and their corresponding labels
            indices = random.sample(range(len(images)), num_images)
            random_images = [images[i] for i in indices]
            random_labels = [
                [round(x, 2) for x in label] if isinstance(label, list) else round(label, 2)
                for label in [task_labels[task][i] for i in indices]
            ]

            # Plot the images with enhanced styling
            fig, axes = plt.subplots(1, num_images, figsize=(15, 9))
            fig.suptitle(f"Training Dataset: {task.upper()} Task", fontsize=16, y=1.05)
            
            # Add background color
            fig.patch.set_facecolor('#f0f0f0')
            
            for i, (img_path, label) in enumerate(zip(random_images, random_labels)):
                img_path_full = os.path.join(main_output_dir, img_path)
                img = Image.open(img_path_full)
                
                if not isinstance(axes, np.ndarray):
                    axes = [axes]
                
                axes[i].imshow(img, cmap="gray")
                axes[i].axis("off")
                axes[i].set_title(f"Sample {i+1}\nLabel: {label}", 
                                bbox=dict(facecolor='white', alpha=0.8),
                                pad=10)
                
                # Add border to each subplot
                for spine in axes[i].spines.values():
                    spine.set_edgecolor('gray')
                    spine.set_linewidth(2)

            # Display the corresponding prompt below the images
            prompt_text = task_prompts[task]
            plt.figtext(0.5, 0.1, prompt_text, 
                    wrap=True, 
                    horizontalalignment='center', 
                    fontsize=18,
                    bbox=dict(facecolor='white', 
                            edgecolor='gray',
                            alpha=0.8,
                            pad=10))
            
            plt.tight_layout(rect=[0, 0.25, 1, 0.95])
            plt.show()
            print("\n")


""" GENERATE EXP3 DATASET """

def generate_dataset_EXP3(main_output_dir="finetuning-EXP3-5000-test", images_per_task=5000):

    # Define tasks with associated questions
    tasks = {

        'type1': "In the grouped bar chart, compare the heights of the two marked bars. Estimate the ratio of the height of the shorter marked bar to the height of the taller marked bar. Use a scale from 0 to 1, where 1 indicates that both marked bars are of equal height. No explaination.",
        'type2': "In the divided stacked bar chart, compare the heights of the two marked segments in the left and right bars. Estimate the ratio of the height of the shorter marked segment to the taller marked segment. Use a scale from 0 to 1, where 1 indicates that both segments are of equal height. No explaination.",
        'type3': "In the mixed grouped bar chart, compare the heights of the two marked bars. Estimate the ratio of the shorter marked bar’s height to the taller marked bar’s height. Use a scale from 0 to 1, where 1 indicates equal height. No explaination.",
        'type4': "In the divided stacked bars, compare the lengths of the two marked segments in the left and right bars. Estimate the ratio of the shorter marked segment’s length to the length of the taller marked segment. Use a scale from 0 to 1, where 1 indicates equal length. No explanation.",
        'type5': "In the complex divided stacked bar chart, compare the lengths of the two marked segments in the left bar. Estimate the ratio of the length of the shorter marked segment to the length of the taller marked segment. Use a scale from 0 to 1, where 1 indicates that both segments are of equal length."

    }

    # Define the target number of images for each task in the dataset
    train_target = images_per_task
    val_target = images_per_task // 5  # 20% of training size for validation
    test_target = images_per_task // 10  # 10% of training size for testing

    # Initialize dataset lists
    combined_dataset_training = []
    combined_dataset_validation = []
    combined_dataset_testing = []

    # Create directories if they don't exist
    image_output_dir = os.path.join(main_output_dir, "images")
    os.makedirs(image_output_dir, exist_ok=True)

    # Global label tracking for uniqueness across datasets
    train_labels = []
    val_labels = []
    test_labels = []

    for task, question in tasks.items():
        print(f"\n--- Generating images for task: {task} ---")

        # Initialize counters for each dataset
        train_counter = 0
        val_counter = 0
        test_counter = 0
        all_counter = 0

        while train_counter < train_target or val_counter < val_target or test_counter < test_target:
            all_counter += 1  # Track total iterations for debugging

            # Generate image and label using the custom module
            image_array, label = L.GPImage.figure4(task)

            # Determine which dataset the label belongs to
            pot = np.random.choice(3)
            if label in train_labels:
                pot = 0
            elif label in val_labels:
                pot = 1
            elif label in test_labels:
                pot = 2

            # Training dataset
            if pot == 0 and train_counter < train_target:
                if label not in train_labels:
                    train_labels.append(label)
                process_and_save_image(image_array, label, question, combined_dataset_training, image_output_dir, task)
                train_counter += 1

            # Validation dataset
            elif pot == 1 and val_counter < val_target:
                if label not in val_labels:
                    val_labels.append(label)
                process_and_save_image(image_array, label, question, combined_dataset_validation, image_output_dir, task)
                val_counter += 1

            # Test dataset
            elif pot == 2 and test_counter < test_target:
                if label not in test_labels:
                    test_labels.append(label)
                process_and_save_image(image_array, label, question, combined_dataset_testing, image_output_dir, task)
                test_counter += 1

        print(f"Task {task} generation completed with {all_counter} iterations.")

    # Save datasets to JSON files
    datasets = {
        "train": combined_dataset_training,
        "val": combined_dataset_validation,
        "test": combined_dataset_testing
    }
    save_datasets_to_json(datasets, main_output_dir)

def display_training_samples_exp3(main_output_dir="finetuning-EXP1-5000-test", num_images=2):
    """
    Display a specified number of random images for each task from the training dataset
    for bar comparison tasks.

    Parameters:
        main_output_dir (str): Directory where datasets and images are stored.
        num_images (int): Number of random images to display for each task.

    Returns:
        None
    """
    import os
    import json
    import random
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt

    # Define tasks and prompts for bar comparisons
    task_prompts = {
        'type1': "In the grouped bar chart, compare the heights of the two marked bars. Estimate the ratio of the height of the shorter marked bar to the height of the taller marked bar. Use a scale from 0 to 1, where 1 indicates that both marked bars are of equal height. No explaination.",
        'type2': "In the divided stacked bar chart, compare the heights of the two marked segments in the left and right bars. Estimate the ratio of the height of the shorter marked segment to the taller marked segment. Use a scale from 0 to 1, where 1 indicates that both segments are of equal height. No explaination.",
        'type3': "In the mixed grouped bar chart, compare the heights of the two marked bars. Estimate the ratio of the shorter marked bar's height to the taller marked bar's height. Use a scale from 0 to 1, where 1 indicates equal height. No explaination.",
        'type4': "In the divided stacked bars, compare the lengths of the two marked segments in the left and right bars. Estimate the ratio of the shorter marked segment's length to the length of the taller marked segment. Use a scale from 0 to 1, where 1 indicates equal length. No explanation.",
        'type5': "In the complex divided stacked bar chart, compare the lengths of the two marked segments in the left bar. Estimate the ratio of the length of the shorter marked segment to the length of the taller marked segment. Use a scale from 0 to 1, where 1 indicates that both segments are of equal length."
    }

    # Define dataset file paths
    json_output_dir = os.path.join(main_output_dir, "json")
    train_file = os.path.join(json_output_dir, "train_dataset.json")

    # Load datasets
    with open(train_file, 'r') as f:
        train_dataset = json.load(f)

    # Group datasets for display
    datasets = {
        "Training": train_dataset,
    }

    # Display random images for each dataset and task
    for dataset_name, dataset in datasets.items():
        print("\n" + "="*70)
        print(f"Displaying random images for {dataset_name} dataset:")
        print("="*70)

        # Add debug print to check tasks
        print("Available tasks in dataset:", set(entry['task'] for entry in dataset))

        # Group images and labels by task
        task_images = {f"type{i}": [] for i in range(1, 6)}
        task_labels = {f"type{i}": [] for i in range(1, 6)}

        for entry in dataset:
            if entry['task'] in task_images:  # Check if task exists in our dictionary
                task_images[entry['task']].append(entry['image'])
                task_labels[entry['task']].append(entry['value'])

        # Display samples for each task
        for task, images in task_images.items():
            if len(images) < num_images:
                print(f"Not enough images for task '{task}' in training dataset.")
                continue

            print(f"\nTask: {task.upper()}")
            print(f"Number of images available for {task}: {len(images)}")
            print("-"*50)

            # Select random images and their corresponding labels
            indices = random.sample(range(len(images)), num_images)
            random_images = [images[i] for i in indices]
            random_labels = [task_labels[task][i] for i in indices]

            # Plot the images with enhanced styling
            fig, axes = plt.subplots(1, num_images, figsize=(15, 7))
            
            # Adjust figure size and spacing
            plt.subplots_adjust(top=0.85, bottom=0.2)  # Adjust these values as needed
            fig.suptitle(f"Training Dataset: {task.upper()}", fontsize=16, y=0.98)
            
            # Add background color
            fig.patch.set_facecolor('#f0f0f0')
            
            for i, (img_path, label) in enumerate(zip(random_images, random_labels)):
                img_path_full = os.path.join(main_output_dir, img_path)
                img = Image.open(img_path_full)
                
                if not isinstance(axes, np.ndarray):
                    axes = [axes]
                
                # Convert image to array and get aspect ratio
                img_array = np.array(img)
                aspect_ratio = img_array.shape[0] / img_array.shape[1]
                
                axes[i].imshow(img_array, cmap="gray", aspect='equal')
                axes[i].axis("off")
                axes[i].set_title(f"Sample {i+1}\nLabel: {label:.2f}", 
                                bbox=dict(facecolor='white', alpha=0.8),
                                pad=10)
                
                # Add border to each subplot
                for spine in axes[i].spines.values():
                    spine.set_edgecolor('gray')
                    spine.set_linewidth(2)

            # Adjust layout before adding prompt
            plt.tight_layout(rect=[0, 0.2, 1, 0.95])  # Adjust these values as needed

            # Display the corresponding prompt below the images
            prompt_text = task_prompts[task]
            plt.figtext(0.5, 0.02, prompt_text, 
                    wrap=True, 
                    horizontalalignment='center', 
                    fontsize=20,
                    bbox=dict(facecolor='white', 
                            edgecolor='gray',
                            alpha=0.8,
                            pad=10))
            
            plt.show()
            print("\n")


""" GENERATE EXP4 DATASET """

def generate_dataset_EXP4(main_output_dir="finetuning-EXP4-5000-test", images_per_task=5000):

    # Define tasks with associated questions
    tasks = {

        "framed": "Estimate the lengths of the two bars with framing. Both lengths should fall between 49 and 60 pixels. No explanation. Format of the answer [xx, xx]",
        "unframed": "Estimate the lengths of the two bars without framing. Both lengths should fall between 49 and 60 pixels. No explanation. Format of the answer [xx, xx]"
    }

    # Define the target number of images for each task in the dataset
    train_target = images_per_task
    val_target = images_per_task // 5  # 20% of training size for validation
    test_target = images_per_task // 10  # 10% of training size for testing

    # Initialize dataset lists
    combined_dataset_training = []
    combined_dataset_validation = []
    combined_dataset_testing = []

    # Create directories if they don't exist
    image_output_dir = os.path.join(main_output_dir, "images")
    os.makedirs(image_output_dir, exist_ok=True)

    # Global label tracking for uniqueness across datasets
    train_labels = []
    val_labels = []
    test_labels = []

    for task, question in tasks.items():
        print(f"\n--- Generating images for task: {task} ---")

        # Initialize counters for each dataset
        train_counter = 0
        val_counter = 0
        test_counter = 0
        all_counter = 0

        while train_counter < train_target or val_counter < val_target or test_counter < test_target:
            all_counter += 1  # Track total iterations for debugging

            # Generate image and label using the custom module
            image_array, label = L.GPImage.figure12(task)

            # Determine which dataset the label belongs to
            pot = np.random.choice(3)
            if label in train_labels:
                pot = 0
            elif label in val_labels:
                pot = 1
            elif label in test_labels:
                pot = 2

            # Training dataset
            if pot == 0 and train_counter < train_target:
                if label not in train_labels:
                    train_labels.append(label)
                process_and_save_image(image_array, label, question, combined_dataset_training, image_output_dir, task)
                train_counter += 1

            # Validation dataset
            elif pot == 1 and val_counter < val_target:
                if label not in val_labels:
                    val_labels.append(label)
                process_and_save_image(image_array, label, question, combined_dataset_validation, image_output_dir, task)
                val_counter += 1

            # Test dataset
            elif pot == 2 and test_counter < test_target:
                if label not in test_labels:
                    test_labels.append(label)
                process_and_save_image(image_array, label, question, combined_dataset_testing, image_output_dir, task)
                test_counter += 1

        print(f"Task {task} generation completed with {all_counter} iterations.")

    # Save datasets to JSON files
    datasets = {
        "train": combined_dataset_training,
        "val": combined_dataset_validation,
        "test": combined_dataset_testing
    }
    save_datasets_to_json(datasets, main_output_dir)

def display_training_samples_exp4(main_output_dir="finetuning-EXP4-5000-test", num_images=2):
    """
    Display a specified number of random images for framed and unframed bar tasks
    from the training dataset.

    Parameters:
        main_output_dir (str): Directory where datasets and images are stored.
        num_images (int): Number of random images to display for each task.

    Returns:
        None
    """
    import os
    import json
    import random
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt

    # Define tasks with associated questions
    tasks = {
        "framed": "Estimate the lengths of the two bars with framing. Both lengths should fall between 49 and 60 pixels. No explanation. Format of the answer [xx, xx]",
        "unframed": "Estimate the lengths of the two bars without framing. Both lengths should fall between 49 and 60 pixels. No explanation. Format of the answer [xx, xx]"
    }

    # Define dataset file paths
    json_output_dir = os.path.join(main_output_dir, "json")
    train_file = os.path.join(json_output_dir, "train_dataset.json")

    # Load datasets
    with open(train_file, 'r') as f:
        train_dataset = json.load(f)

    # Group datasets for display
    datasets = {
        "Training": train_dataset,
    }

    # Display random images for each dataset and task
    for dataset_name, dataset in datasets.items():
        print("\n" + "="*70)
        print(f"Displaying random images for {dataset_name} dataset:")
        print("="*70)

        # Group images and labels by task
        task_images = {"framed": [], "unframed": []}
        task_labels = {"framed": [], "unframed": []}

        for entry in dataset:
            if entry['task'] in tasks:
                task_images[entry['task']].append(entry['image'])
                task_labels[entry['task']].append(entry['value'])

        # Display samples for each task
        for task, images in task_images.items():
            if len(images) < num_images:
                print(f"Not enough images for task '{task}' in training dataset.")
                continue

            print(f"\nTask: {task.upper()}")
            print(f"Number of images available: {len(images)}")
            print("-"*50)

            # Select random images and their corresponding labels
            indices = random.sample(range(len(images)), num_images)
            random_images = [images[i] for i in indices]
            random_labels = [task_labels[task][i] for i in indices]

            # Plot the images with enhanced styling
            fig, axes = plt.subplots(1, num_images, figsize=(15, 7))
            
            # Adjust figure size and spacing
            plt.subplots_adjust(top=0.85, bottom=0.2)
            fig.suptitle(f"Training Dataset: {task.upper()}", fontsize=16, y=0.98)
            
            # Add background color
            fig.patch.set_facecolor('#f0f0f0')
            
            for i, (img_path, label) in enumerate(zip(random_images, random_labels)):
                img_path_full = os.path.join(main_output_dir, img_path)
                img = Image.open(img_path_full)
                
                if not isinstance(axes, np.ndarray):
                    axes = [axes]
                
                # Convert image to array and ensure proper display
                img_array = np.array(img)
                axes[i].imshow(img_array, cmap="gray", aspect='equal')
                axes[i].axis("off")
                # Format label as [xx, xx]
                label_str = f"[{label[0]:.1f}, {label[1]:.1f}]" if isinstance(label, list) else str(label)
                axes[i].set_title(f"Sample {i+1}\nLabel: {label_str}", 
                                bbox=dict(facecolor='white', alpha=0.8),
                                pad=10)
                
                # Add border to each subplot
                for spine in axes[i].spines.values():
                    spine.set_edgecolor('gray')
                    spine.set_linewidth(2)

            # Adjust layout before adding prompt
            plt.tight_layout(rect=[0, 0.2, 1, 0.95])

            # Display the corresponding prompt below the images
            prompt_text = tasks[task]
            plt.figtext(0.5, 0.02, prompt_text, 
                    wrap=True, 
                    horizontalalignment='center', 
                    fontsize=20,
                    bbox=dict(facecolor='white', 
                            edgecolor='gray',
                            alpha=0.8,
                            pad=10))
            
            plt.show()
            print("\n")

""" GENERATE EXP5 DATASET """

def generate_dataset_EXP5(main_output_dir="finetuning-EXP5-5000-test", images_per_task=5000):

    # Define tasks with associated questions
    tasks = {
        "10": "Please estimate how many dots were added to the initial 10 dots. The answer must be within the range of 1 to 10. Number only. No explanation.",
        "100": "Please estimate how many dots were added to the initial 100 dots. The answer must be within the range of 1 to 10. Number only. No explanation.",
        "1000": "Please estimate how many dots were added to the initial 1000 dots. The answer must be within the range of 1 to 10. Number only. No explanation."
    }

    # Define the target number of images for each task in the dataset
    train_target = images_per_task
    val_target = images_per_task // 5  # 20% of training size for validation
    test_target = images_per_task // 10  # 10% of training size for testing

    # Initialize dataset lists
    combined_dataset_training = []
    combined_dataset_validation = []
    combined_dataset_testing = []

    # Create directories if they don't exist
    image_output_dir = os.path.join(main_output_dir, "images")
    os.makedirs(image_output_dir, exist_ok=True)

    # Global label tracking for uniqueness across datasets
    train_labels = []
    val_labels = []
    test_labels = []

    for task, question in tasks.items():
        print(f"\n--- Generating images for task: {task} ---")

        # Initialize counters for each dataset
        train_counter = 0
        val_counter = 0
        test_counter = 0
        all_counter = 0

        while train_counter < train_target or val_counter < val_target or test_counter < test_target:
            all_counter += 1  # Track total iterations for debugging

            # Generate image and label using the custom module
            image_array, label = L.GPImage.weber(task)

            # Determine which dataset the label belongs to
            pot = np.random.choice(3)
            if label in train_labels:
                pot = 0
            elif label in val_labels:
                pot = 1
            elif label in test_labels:
                pot = 2

            # Training dataset
            if pot == 0 and train_counter < train_target:
                if label not in train_labels:
                    train_labels.append(label)
                process_and_save_image(image_array, label, question, combined_dataset_training, image_output_dir, task)
                train_counter += 1

            # Validation dataset
            elif pot == 1 and val_counter < val_target:
                if label not in val_labels:
                    val_labels.append(label)
                process_and_save_image(image_array, label, question, combined_dataset_validation, image_output_dir, task)
                val_counter += 1

            # Test dataset
            elif pot == 2 and test_counter < test_target:
                if label not in test_labels:
                    test_labels.append(label)
                process_and_save_image(image_array, label, question, combined_dataset_testing, image_output_dir, task)
                test_counter += 1

        print(f"Task {task} generation completed with {all_counter} iterations.")

    # Save datasets to JSON files
    datasets = {
        "train": combined_dataset_training,
        "val": combined_dataset_validation,
        "test": combined_dataset_testing
    }
    save_datasets_to_json(datasets, main_output_dir)


def display_training_samples_exp5(main_output_dir="finetuning-EXP1-5000-test", num_images=2):
    """
    Display a specified number of random images for dot estimation tasks
    from the training dataset.

    Parameters:
        main_output_dir (str): Directory where datasets and images are stored.
        num_images (int): Number of random images to display for each task.

    Returns:
        None
    """
    import os
    import json
    import random
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt

    # Define tasks with associated questions
    tasks = {
        "10": "Please estimate how many dots were added to the initial 10 dots. The answer must be within the range of 1 to 10. Number only. No explanation.",
        "100": "Please estimate how many dots were added to the initial 100 dots. The answer must be within the range of 1 to 10. Number only. No explanation.",
        "1000": "Please estimate how many dots were added to the initial 1000 dots. The answer must be within the range of 1 to 10. Number only. No explanation."
    }

    # Define dataset file paths
    json_output_dir = os.path.join(main_output_dir, "json")
    train_file = os.path.join(json_output_dir, "train_dataset.json")

    # Load datasets
    with open(train_file, 'r') as f:
        train_dataset = json.load(f)

    # Group datasets for display
    datasets = {
        "Training": train_dataset,
    }

    # Display random images for each dataset and task
    for dataset_name, dataset in datasets.items():
        print("\n" + "="*70)
        print(f"Displaying random images for {dataset_name} dataset:")
        print("="*70)

        # Group images and labels by task
        task_images = {"10": [], "100": [], "1000": []}
        task_labels = {"10": [], "100": [], "1000": []}

        for entry in dataset:
            if entry['task'] in tasks:
                task_images[entry['task']].append(entry['image'])
                task_labels[entry['task']].append(entry['value'])

        # Display samples for each task
        for task, images in task_images.items():
            if len(images) < num_images:
                print(f"Not enough images for task '{task}' in training dataset.")
                continue

            print(f"\nTask: Base {task} dots")
            print(f"Number of images available: {len(images)}")
            print("-"*50)

            # Select random images and their corresponding labels
            indices = random.sample(range(len(images)), num_images)
            random_images = [images[i] for i in indices]
            random_labels = [task_labels[task][i] for i in indices]

            # Plot the images with enhanced styling
            fig, axes = plt.subplots(1, num_images, figsize=(15, 7))
            
            # Adjust figure size and spacing
            plt.subplots_adjust(top=0.85, bottom=0.2)
            fig.suptitle(f"Training Dataset: Base {task} Dots", fontsize=16, y=0.98)
            
            # Add background color
            fig.patch.set_facecolor('#f0f0f0')
            
            for i, (img_path, label) in enumerate(zip(random_images, random_labels)):
                img_path_full = os.path.join(main_output_dir, img_path)
                img = Image.open(img_path_full)
                
                if not isinstance(axes, np.ndarray):
                    axes = [axes]
                
                # Convert image to array and ensure proper display
                img_array = np.array(img)
                axes[i].imshow(img_array, cmap="gray", aspect='equal')
                axes[i].axis("off")
                axes[i].set_title(f"Sample {i+1}\nAdded dots: {label:.0f}", 
                                bbox=dict(facecolor='white', alpha=0.8),
                                pad=10)
                
                # Add border to each subplot
                for spine in axes[i].spines.values():
                    spine.set_edgecolor('gray')
                    spine.set_linewidth(2)

            # Adjust layout before adding prompt
            plt.tight_layout(rect=[0, 0.2, 1, 0.95])

            # Display the corresponding prompt below the images
            prompt_text = tasks[task]
            plt.figtext(0.5, 0.02, prompt_text, 
                    wrap=True, 
                    horizontalalignment='center', 
                    fontsize=20,
                    bbox=dict(facecolor='white', 
                            edgecolor='gray',
                            alpha=0.8,
                            pad=10))
            
            plt.show()
            print("\n")


""" Use for all experiments"""

def process_and_save_image(image_array, label, question, dataset, output_dir, task):
    """
    Process the image, add noise, save it as a file, and append it to the dataset.
    """
    # Add noise to the image
    image_array = image_array.astype(np.float32)
    noise_mask = (image_array == 0)
    noise = np.random.uniform(0, 0.05, image_array.shape)
    image_array[noise_mask] += noise[noise_mask]

    # Convert to uint8 and save the image
    image_array_uint8 = (image_array * 255).astype(np.uint8)
    pil_image = Image.fromarray(image_array_uint8)

    unique_id = str(uuid.uuid4())
    image_filename = os.path.join(output_dir, f"{task}_{unique_id}.jpg")
    pil_image.save(image_filename)

    # Ensure `label` is serializable (convert NumPy data types to native Python types)
    if isinstance(label, list):
        label = [float(x) for x in label]  # Ensure elements are Python floats
    else:
        label = float(label)

    # Create a JSON-friendly entry
    json_entry = {
        'id': unique_id,
        'image': f"images/{task}_{unique_id}.jpg",
        "task": task,
        'question': question,
        'value': label
    }
    dataset.append(json_entry)
    return json_entry


def save_datasets_to_json(datasets, output_dir):
    """
    Save the datasets to JSON files.
    """
    json_output_dir = os.path.join(output_dir, "json")
    os.makedirs(json_output_dir, exist_ok=True)

    for dataset_type, dataset in datasets.items():
        filename = f"{dataset_type}_dataset.json"
        filepath = os.path.join(json_output_dir, filename)

        # Debugging: Print an example entry
        if dataset:
            print(f"Saving {dataset_type} dataset. Example entry:", dataset[0])

        with open(filepath, 'w') as json_file:
            try:
                json.dump(dataset, json_file, indent=4, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o)
                print(f"{dataset_type.capitalize()} dataset saved as '{filename}' in '{json_output_dir}'")
            except TypeError as e:
                print(f"Error saving {dataset_type} dataset: {e}")
                for i, entry in enumerate(dataset):
                    print(f"Entry {i}:", entry)  # Print problematic entries
                raise


def verify_dataset(main_output_dir="finetuning-EXP2-5000-test"):
    """
    Verify the dataset by checking the number of images for each task,
    ensuring uniqueness across datasets, and identifying overlaps.

    Parameters:
        main_output_dir (str): Directory where datasets and images are saved.

    Returns:
        None
    """
    import os
    import json
    from collections import Counter

    # Define dataset file paths
    json_output_dir = os.path.join(main_output_dir, "json")
    train_file = os.path.join(json_output_dir, "train_dataset.json")
    val_file = os.path.join(json_output_dir, "val_dataset.json")
    test_file = os.path.join(json_output_dir, "test_dataset.json")

    # Load datasets
    with open(train_file, 'r') as f:
        train_dataset = json.load(f)
    with open(val_file, 'r') as f:
        val_dataset = json.load(f)
    with open(test_file, 'r') as f:
        test_dataset = json.load(f)

    # Print number of images for each dataset with better formatting
    print("\n" + "="*50)
    print(" "*15 + "DATASET SUMMARY")
    print("="*50)
    print(f"\n📊 Total Images per Dataset:")
    print("-" * 40)
    print(f"🔵 Training:    {len(train_dataset):,} images")
    print(f"🟡 Validation:  {len(val_dataset):,} images")
    print(f"🟢 Testing:     {len(test_dataset):,} images")
    print(f"📈 Total:       {len(train_dataset) + len(val_dataset) + len(test_dataset):,} images")

    # Function to count images per task in a dataset
    def count_images_per_task(dataset):
        return Counter(entry['task'] for entry in dataset)

    # Count images for each task
    train_task_count = count_images_per_task(train_dataset)
    val_task_count = count_images_per_task(val_dataset)
    test_task_count = count_images_per_task(test_dataset)

    # Print task-wise image counts with enhanced formatting
    print("\n" + "="*50)
    print(" "*15 + "TASK DISTRIBUTION")
    print("="*50)
    print("\n| Dataset      | Task                   | Image Count |")
    print("|" + "-"*12 + "|" + "-"*24 + "|" + "-"*13 + "|")
    
    # Print tasks with color indicators
    for task, count in train_task_count.items():
        print(f"| 🔵 Training | {task:<22} | {count:>11} |")
    for task, count in val_task_count.items():
        print(f"| 🟡 Valid    | {task:<22} | {count:>11} |")
    for task, count in test_task_count.items():
        print(f"| 🟢 Test     | {task:<22} | {count:>11} |")

    # Extract unique labels for each dataset
    train_labels = {tuple(entry['value']) if isinstance(entry['value'], list) else entry['value'] for entry in train_dataset}
    val_labels = {tuple(entry['value']) if isinstance(entry['value'], list) else entry['value'] for entry in val_dataset}
    test_labels = {tuple(entry['value']) if isinstance(entry['value'], list) else entry['value'] for entry in test_dataset}

    # Check for overlaps
    train_val_overlap = train_labels & val_labels
    train_test_overlap = train_labels & test_labels
    val_test_overlap = val_labels & test_labels

    # Print overlap information with enhanced formatting
    print("\n" + "="*50)
    print(" "*15 + "OVERLAP ANALYSIS")
    print("="*50)
    
    if train_val_overlap:
        print(f"⚠️  Overlap between Training and Validation: {len(train_val_overlap)} entries")
    else:
        print("✅ No overlap between Training and Validation")

    if train_test_overlap:
        print(f"⚠️  Overlap between Training and Testing: {len(train_test_overlap)} entries")
    else:
        print("✅ No overlap between Training and Testing")

    if val_test_overlap:
        print(f"⚠️  Overlap between Validation and Testing: {len(val_test_overlap)} entries")
    else:
        print("✅ No overlap between Validation and Testing")

    # Print unique labels report with enhanced formatting
    print("\n" + "="*50)
    print(" "*15 + "UNIQUE LABELS")
    print("="*50)
    print("\n| Dataset      | Total Unique Labels |")
    print("|" + "-"*12 + "|" + "-"*19 + "|")
    print(f"| 🔵 Training | {len(train_labels):>17} |")
    print(f"| 🟡 Valid    | {len(val_labels):>17} |")
    print(f"| 🟢 Test     | {len(test_labels):>17} |")
    print("\n" + "="*50)