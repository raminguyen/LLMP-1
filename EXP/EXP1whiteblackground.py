import os
import sys
import math
import numpy as np
import pandas as pd
import svgwrite
import skimage.draw
import matplotlib.pyplot as plt
from PIL import Image
import fitz  # PyMuPDF
from sklearn.metrics import mean_absolute_error
from svgpathtools import svg2paths
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path
import cairosvg
import seaborn as sns
import tempfile
import seaborn as sns
import matplotlib.pyplot as plt
# Add LLMP path
sys.path.append('../')
import LLMP as L
import os
import matplotlib.pyplot as plt
from PIL import Image
import fitz  # PyMuPDF for handling PDFs
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd

def display_images_combined_by_degree(folder):
    """
    Display aliased, anti-aliased, and vectorized images for both white and black backgrounds,
    grouped by degree, in a clean and compact layout with properly centered labels.

    Parameters:
        folder (str): Path to the folder containing the images.
    """
    # Collect aliased, anti-aliased, and vectorized files
    aliased_files = sorted([f for f in os.listdir(folder) if "aliased_image" in f and f.endswith(".png")])
    antialiased_files = sorted([f for f in os.listdir(folder) if "antialiased_image" in f and f.endswith(".png")])
    vectorized_files = sorted([f for f in os.listdir(folder) if "vectorized_image" in f and f.endswith(".pdf")])

    # Extract unique identifiers (background + angle)
    aliased_ids = set(f.replace("aliased_image_", "").replace(".png", "") for f in aliased_files)
    antialiased_ids = set(f.replace("antialiased_image_", "").replace(".png", "") for f in antialiased_files)
    vectorized_ids = set(f.replace("vectorized_image_", "").replace(".pdf", "") for f in vectorized_files)

    # Find common IDs
    common_ids = sorted(aliased_ids & antialiased_ids & vectorized_ids)

    # Group by degree and background
    degree_data = {}
    for img_id in common_ids:
        bg_color, angle = img_id.split("_")
        if angle not in degree_data:
            degree_data[angle] = {}
        degree_data[angle][bg_color] = img_id

    # Sort by degree
    sorted_degrees = sorted(degree_data.keys(), key=int)

    # Set up grid for displaying images
    num_rows = len(sorted_degrees)
    fig, axes = plt.subplots(num_rows, 6, figsize=(18, num_rows * 2))
    fig.subplots_adjust(wspace=0.05, hspace=0.2)

    # Add column headers, centered
    column_labels = [
        "Aliased (White)", "Anti-Aliased (White)", "Vectorized (White)",
        "Aliased (Black)", "Anti-Aliased (Black)", "Vectorized (Black)"
    ]
    for col, label in enumerate(column_labels):
        fig.text(
            0.12 + col * 0.155, 
            1.02, 
            label, 
            ha="center", 
            va="center", 
            fontsize=14, 
            fontweight="bold"
        )

    # Iterate through degrees and add images to grid
    for row_idx, degree in enumerate(sorted_degrees):
        # Add degree label on the left, centered vertically
        fig.text(
            0.02, 
            1 - (row_idx + 0.5) / num_rows, 
            f"{degree}°", 
            fontsize=25, 
            fontweight="bold", 
            va="center", 
            ha="center"
        )

        for col_idx, bg_color in enumerate(["white", "black"]):
            if bg_color in degree_data[degree]:
                img_id = degree_data[degree][bg_color]

                # Aliased Image
                aliased_path = os.path.join(folder, f"aliased_image_{img_id}.png")
                img_aliased = Image.open(aliased_path)
                axes[row_idx, col_idx * 3].imshow(img_aliased, cmap="gray")
                axes[row_idx, col_idx * 3].axis("off")

                # Anti-Aliased Image
                antialiased_path = os.path.join(folder, f"antialiased_image_{img_id}.png")
                img_antialiased = Image.open(antialiased_path).convert("L")
                axes[row_idx, col_idx * 3 + 1].imshow(img_antialiased, cmap="gray")
                axes[row_idx, col_idx * 3 + 1].axis("off")

                # Vectorized Image
                vectorized_path = os.path.join(folder, f"vectorized_image_{img_id}.pdf")
                pdf_document = fitz.open(vectorized_path)
                page = pdf_document[0]
                pix = page.get_pixmap(dpi=300)
                img_vectorized = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img_vectorized = img_vectorized.resize((img_aliased.width, img_aliased.height), Image.LANCZOS)
                axes[row_idx, col_idx * 3 + 2].imshow(img_vectorized, cmap="gray")
                axes[row_idx, col_idx * 3 + 2].axis("off")
                pdf_document.close()

    plt.tight_layout(rect=[0.05, 0, 1, 0.98])  # Adjust layout
    plt.show()



def clean_experiment_data(file_path):
    """
    Cleans and processes experiment results from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file containing experiment results.

    Returns:
        pd.DataFrame: A cleaned DataFrame with extracted and formatted data.
    """
    # Set pandas display options for easier debugging (optional)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    # Read the CSV file
    df = pd.read_csv(file_path)

    # Clean the 'prediction' column: strip whitespace, newline characters, and ensure numeric type
    df['prediction'] = df['prediction'].astype(str).str.strip().str.replace("\n", "")
    df['prediction'] = pd.to_numeric(df['prediction'], errors='coerce')  # Convert to float, NaN if invalid

    # Extract the file name from the file path
    df['file_name'] = df['file_path'].str.split('/').str[-1]

    # Reorder and rename columns for clarity
    df = df[['file_name', 'ground_truth', 'model', 'prediction']]

    return df



def calculate_mlae(gt, answers):
    """
    Calculate Mean Log Absolute Error (MLAE), handling NaN values.

    Parameters:
        gt (list or np.ndarray): Ground truth values.
        answers (np.ndarray): Model predictions.

    Returns:
        float: The calculated MLAE, or NaN if no valid data exists.
    """
    # Convert to numpy arrays
    gt = np.array(gt, dtype=np.float32)
    answers = np.array(answers, dtype=np.float32)

    # Remove NaN values
    valid_mask = ~np.isnan(answers)
    gt_filtered = gt[valid_mask]
    answers_filtered = answers[valid_mask]

    if len(answers_filtered) == 0:
        return np.nan

    # Calculate MLAE
    return np.log2(mean_absolute_error(gt_filtered, answers_filtered) + 0.125)


def calculate_row_mlae(row):
    """
    Calculate MLAE for a single row using its ground truth and prediction.

    Parameters:
        row (pd.Series): A single row of the DataFrame.

    Returns:
        float: The MLAE for the row.
    """
    # Extract ground truth and prediction from the row
    ground_truth = row['ground_truth']
    prediction = row['prediction']

    # Convert values to arrays for compatibility with `calculate_mlae`
    gt = [ground_truth]
    answers = [prediction]

    # Calculate MLAE for the row
    return calculate_mlae(gt, answers)


def plot_mlae_heatmap(df):
    """
    Plot a heatmap showing MLAE values for each model and image.

    Parameters:
        df (pd.DataFrame): DataFrame containing MLAE values with columns 
                           'file_name', 'model', and 'MLAE'.

    Returns:
        None
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Pivot the DataFrame for heatmap data
    heatmap_data = df.pivot(index='file_name', columns='model', values='MLAE')

    # Check for missing values in the pivoted data
    if heatmap_data.isnull().values.any():
        print("Warning: Heatmap contains NaN values. Consider cleaning the data.")

    # Plot the heatmap
    plt.figure(figsize=(16, 12))

    sns.heatmap(
        heatmap_data,
        annot=True,                # Annotate cells with MLAE values
        cmap='coolwarm',           # Color map for heatmap
        fmt=".2f",                 # Format values with 2 decimals
        cbar_kws={'label': 'MLAE'},  # Label for color bar
        linewidths=0.5,            # Add gridlines between cells
        annot_kws={"fontsize": 10} # Font size for annotations
    )

    plt.title('MLAE Heatmap for Models and Images', fontsize=16)
    plt.xlabel('Model Name', fontsize=14)
    plt.ylabel('Image Name', fontsize=14)

    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=10)

    plt.tight_layout()
    plt.show()


#### 

""" Generate random angle and vectorized (PDF), aliased (PNG), and anti-alised (PNG) images """

####

def calculate_end_point(center, angle, length):
    """
    Calculate the end point of a line given a center point, angle (in degrees), and length.

    Parameters:
        center (tuple): Center point of the canvas (x, y).
        angle (float): Angle of the line in degrees.
        length (int): Length of the line.

    Returns:
        tuple: (end_x, end_y) coordinates.
    """
    angle_rad = np.radians(angle)  # Convert angle to radians
    end_x = center[0] + length * np.cos(angle_rad)
    end_y = center[1] - length * np.sin(angle_rad)  # Subtract for correct y-axis direction
    return end_x, end_y


def generate_vectorized_image_to_pdf(center, first_angle, second_angle, line_length, canvas_size, pdf_file, bg_color="white"):
    """
    Generate a vectorized SVG in memory and directly convert it to a PDF using cairosvg.
    """
    import io

    # Create an SVG drawing in memory
    dwg = svgwrite.Drawing(size=(canvas_size, canvas_size))
    dwg.add(dwg.rect(insert=(0, 0), size=(canvas_size, canvas_size), fill=bg_color))
    
    # Calculate end points for the two lines
    end_point_1 = calculate_end_point(center, first_angle, line_length)
    end_point_2 = calculate_end_point(center, second_angle, line_length)
    
    # Line color
    line_color = "white" if bg_color == "black" else "black"
    
    # Draw the two lines
    dwg.add(dwg.line(start=center, end=end_point_1, stroke=line_color, stroke_width=2))
    dwg.add(dwg.line(start=center, end=end_point_2, stroke=line_color, stroke_width=2))
    
    # Convert SVG content to PDF
    svg_content = io.BytesIO()
    svg_content.write(dwg.tostring().encode('utf-8'))  # Write the SVG content as bytes
    svg_content.seek(0)  # Reset the stream position

    try:
        cairosvg.svg2pdf(file_obj=svg_content, write_to=pdf_file)
        print(f"PDF saved: {pdf_file}")
    except Exception as e:
        print(f"Error converting SVG to PDF: {e}")


def generate_aliased_image(center, first_angle, second_angle, line_length, canvas_size, bg_color="white"):
    """
    Generate an aliased image with consistent background and line properties.

    Parameters:
        center (tuple): Center point of the canvas (x, y).
        first_angle (float): Angle of the first line in degrees.
        second_angle (float): Angle of the second line in degrees.
        line_length (int): Length of each line.
        canvas_size (int): Size of the canvas (square).
        bg_color (str): Background color ("white" or "black").
    
    Returns:
        np.ndarray: Aliased image.
    """
    import skimage.draw

    # Set background and line colors
    if bg_color == "white":
        image = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255  # White background
        line_color = 0  # Black lines
    else:
        image = np.zeros((canvas_size, canvas_size), dtype=np.uint8)  # Black background
        line_color = 255  # White lines

    # Extract the center coordinates
    center_y, center_x = int(center[1]), int(center[0])

    # First line
    end_x1, end_y1 = calculate_end_point(center, first_angle, line_length)
    rr, cc = skimage.draw.line(center_y, center_x, int(round(end_y1)), int(round(end_x1)))
    image[rr, cc] = line_color

    # Second line
    end_x2, end_y2 = calculate_end_point(center, second_angle, line_length)
    rr, cc = skimage.draw.line(center_y, center_x, int(round(end_y2)), int(round(end_x2)))
    image[rr, cc] = line_color

    return image


def generate_antialiased_image(center, first_angle, second_angle, line_length, canvas_size, bg_color="white"):
    """
    Generate an anti-aliased image using consistent parameters.
    """
    # Set background and line colors
    if bg_color == "white":
        image = np.ones((canvas_size, canvas_size), dtype=float)  # Black background
        line_color = -1.0  # black lines
    else:
        image = np.zeros((canvas_size, canvas_size), dtype=float)  # White background
        line_color = 1.0  # white lines

    # First line
    end_x1, end_y1 = calculate_end_point(center, first_angle, line_length)
    rr, cc, val = skimage.draw.line_aa(int(center[1]), int(center[0]), int(end_y1), int(end_x1))
    image[rr, cc] = np.clip(image[rr, cc] + line_color * val, 0, 1)

    # Second line
    end_x2, end_y2 = calculate_end_point(center, second_angle, line_length)
    rr, cc, val = skimage.draw.line_aa(int(center[1]), int(center[0]), int(end_y2), int(end_x2))
    image[rr, cc] = np.clip(image[rr, cc] + line_color * val, 0, 1)

    return image


def convert_svg_to_pdf(svg_file, output_pdf):
    """Convert an SVG file to a PDF file."""
    try:
        print(f"Converting SVG to PDF: {svg_file} -> {output_pdf}")
        cairosvg.svg2pdf(url=svg_file, write_to=output_pdf)
        print(f"PDF saved: {output_pdf}")
    except Exception as e:
        print(f"Error converting SVG to PDF: {svg_file} -> {e}")

def generate_images_for_multiple_angles(canvas_size=100, line_length=30, num_angles=10, output_folder="generated_images"):
    """
    Generate vectorized, aliased, and anti-aliased images for random angles.

    Parameters:
        canvas_size (int): Size of the canvas (square dimensions).
        line_length (int): Length of the lines to draw.
        num_angles (int): Number of random angles to generate.
        output_folder (str): Folder to save the generated images.
    """
    # Generate 10 random angles between 10° and 90°
    angles = np.random.randint(10, 90, num_angles)

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Define background colors
    backgrounds = ["white", "black"]

    for angle in angles:
        print(f"Generating images for angle: {angle}°")
        
        # Consistent parameters for all image types
        center = (canvas_size / 2, canvas_size / 2)
        first_angle = 45
        second_angle = first_angle + angle

        for bg_color in backgrounds:
            # Use consistent parameters
            line_color = "black" if bg_color == "white" else "white"

            # Vectorized Image (PDF)
            pdf_file = os.path.join(output_folder, f"vectorized_image_{bg_color}_{angle}.pdf")
            generate_vectorized_image_to_pdf(center, first_angle, second_angle, line_length, canvas_size, pdf_file, bg_color)

            # Aliased Image
            aliased_image = generate_aliased_image(center, first_angle, second_angle, line_length, canvas_size, bg_color)
            aliased_image_file = os.path.join(output_folder, f"aliased_image_{bg_color}_{angle}.png")
            Image.fromarray(aliased_image).save(aliased_image_file)

            # Anti-Aliased Image
            antialiased_image = generate_antialiased_image(center, first_angle, second_angle, line_length, canvas_size, bg_color)
            antialiased_image_file = os.path.join(output_folder, f"antialiased_image_{bg_color}_{angle}.png")
            Image.fromarray((antialiased_image * 255).astype(np.uint8)).save(antialiased_image_file)


import time 

import os
import numpy as np
from PIL import Image
import fitz
import logging
from datetime import datetime
from typing import List, Tuple, Dict, Any

class Runexp1:
    def __init__(self):
        self.tasks = ["angle"]  # Currently focusing only on the angle task
        self.queries = {
            "angle": "Estimate the angle (range: 0-90 degrees). Number only. No explanation."
        }
        self._setup_logging()

    def _setup_logging(self):
        """Set up logging configuration for the experiment."""
        log_dir = "experiment_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"experiment_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def run_angle_experiment(self, data, model_instances, output_folder="EXP1-Results"):
        """
        Run the 'angle' experiment using the provided data and model instances.

        Parameters:
            data (list): List of tuples (file_path, img_array, ground_truth).
            model_instances (dict): Dictionary of model instances.
            output_folder (str): Directory to save results.

        Returns:
            None
        """
        try:
            print("Starting angle experiment")

            if not data:
                raise ValueError("Input data is empty")

            if not model_instances:
                raise ValueError("Model instances dictionary is empty")

            task_name = "angle"
            query = self.queries[task_name]

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
                print(f"Created output folder: {output_folder}")

            # Log experiment details
            #print(f"Processing {len(data)} images")
            print(f"Using models: {list(model_instances.keys())}")

            results = []

            # Iterate over data and models
            for file_path, img_array, ground_truth in data:
                for model_name, model_instance in model_instances.items():
                    try:
                        print(f"Running model: {model_name} on image: {file_path}")
                        if isinstance(img_array, np.ndarray):
                            prediction = model_instance.query(query, img_array)
                        else:
                            raise ValueError(f"Invalid image format for {file_path}")
                        
                        # Append results
                        results.append({
                            "file_path": file_path,
                            "ground_truth": ground_truth,
                            "model": model_name,
                            "prediction": prediction
                        })
                    except Exception as model_error:
                        print(f"Error processing {file_path} with {model_name}: {str(model_error)}")

            # Save results to CSV
            results_file = os.path.join(output_folder, f"{task_name}_results.csv")
            import csv
            with open(results_file, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=["file_path", "ground_truth", "model", "prediction"])
                writer.writeheader()
                writer.writerows(results)
            
            print(f"Results saved to {results_file}")

        except Exception as e:
            print(f"Experiment failed: {str(e)}")



def load_image(file_path):
    """
    Load an image or convert a PDF to an image. Returns a grayscale NumPy array.

    Parameters:
        file_path (str): Path to the file (image or PDF).

    Returns:
        np.ndarray: Grayscale image as a NumPy array.
    """
    if file_path.endswith(".pdf"):
        try:
            # Open the PDF file using PyMuPDF
            #print(f"Processing PDF file: {file_path}")
            doc = fitz.open(file_path)
            page = doc[0]  # Get the first page
            pix = page.get_pixmap(dpi=300)  # Render at 300 DPI for better quality
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            #print(f"PDF successfully converted to NumPy array: {file_path}")
            return np.array(img.convert("L"))
        except Exception as e:
            raise ValueError(f"Failed to process PDF file {file_path}: {e}")
    else:
        try:
            print(f"Processing image file: {file_path}")
            img = Image.open(file_path).convert("L")
            print(f"Image successfully loaded: {file_path}")
            return np.array(img)
        except Exception as e:
            raise ValueError(f"Failed to process image file {file_path}: {e}")


def prepare_image_data(image_dir):
    """
    Prepare image data from a given directory.

    Parameters:
        image_dir (str): Directory containing the images.

    Returns:
        list: List of tuples (file_path, img_array, ground_truth).
    """
    image_data = []
    for file in os.listdir(image_dir):
        file_path = os.path.join(image_dir, file)
        if file.endswith((".png", ".pdf")):
            try:
                img_array = load_image(file_path)
                parts = file.rsplit("_", maxsplit=2)
                
                if len(parts) >= 3 and parts[-1].split(".")[0].isdigit():
                    ground_truth = int(parts[-1].split(".")[0])  # Extract the angle
                else:
                    ground_truth = None

                image_data.append((file_path, img_array, ground_truth))
            except ValueError as e:
                print(e)
    return image_data



def average_mlae_and_visualize(df):
    """
    Calculate the average MLAE for each image type and model, then visualize the results.

    Parameters:
        df (pd.DataFrame): DataFrame containing MLAE values with columns 
                           'file_name', 'model', 'MLAE'.

    Returns:
        None
    """
    # Extract image type from file name (e.g., "aliased", "antialiased", "vectorized")
    df['image_type'] = df['file_name'].str.extract(r'(aliased|antialiased|vectorized)')
    
    # Group by image type and model, then calculate the average MLAE
    avg_mlae = df.groupby(['image_type', 'model'])['MLAE'].mean().reset_index()
    
    # Pivot the DataFrame for visualization
    pivot_avg_mlae = avg_mlae.pivot(index='image_type', columns='model', values='MLAE')
    
    # Plot grouped bar chart
    pivot_avg_mlae.plot(
        kind='bar',
        figsize=(12, 8),
        colormap='viridis',
        edgecolor='black'
    )
    
    # Customize the plot
    plt.title("Average MLAE by Image Type and Model", fontsize=16, pad=20)
    plt.xlabel("Image Type", fontsize=14)
    plt.ylabel("Average MLAE", fontsize=14)
    plt.xticks(rotation=0, fontsize=12)
    plt.legend(title="Model", fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_vectorized_predictions_scatter(df):
    """
    Creates a scatterplot comparing predictions to ground truth for vectorized images.

    Parameters:
        df (pd.DataFrame): The DataFrame containing prediction data with columns:
                           'file_name', 'ground_truth', 'model', 'prediction'.

    Returns:
        None
    """
    # Filter rows for vectorized images
    vectorized_data = df[df['file_name'].str.contains('vectorized')]

    # Sort by ground truth for consistent ordering
    vectorized_sorted = vectorized_data.sort_values(by='ground_truth')

    # Define models and corresponding styles
    models = ['gpt4o', 'GeminiProVision', 'Gemini1_5Flash']
    colors = ['#FFD700', '#4682B4', '#4B0082']  # Yellow, Teal, Purple
    markers = ['o', 's', '^']

    # Create the scatterplot
    plt.figure(figsize=(12, 8))

    # Plot the ground truth as a baseline
    plt.plot(
        vectorized_sorted['ground_truth'],
        vectorized_sorted['ground_truth'],
        color='orange',
        linestyle='--',
        linewidth=2,
        label='Ground Truth (Reference)'
    )

    # Plot predictions for each model
    for model, color, marker in zip(models, colors, markers):
        model_data = vectorized_sorted[vectorized_sorted['model'] == model]
        plt.scatter(
            model_data['ground_truth'],  # X-axis: Ground Truth
            model_data['prediction'],   # Y-axis: Prediction
            label=f'{model} Prediction',
            color=color,
            marker=marker,
            s=100,  # Marker size
            edgecolor='black'  # Black edges for better visibility
        )

    # Add titles, labels, and legend
    plt.title("Scatterplot: Predictions vs. Ground Truth for Vectorized Images", fontsize=14)
    plt.xlabel("Ground Truth", fontsize=12)
    plt.ylabel("Prediction", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.6, linestyle="--")

    # Ensure layout fits properly
    plt.tight_layout()

    # Display the plot
    plt.show()


def analyze_best_model_by_image_type(df):
    """
    Analyze and determine the best model for each image type based on MLAE and see which models work better.

    Parameters:
        df (pd.DataFrame): DataFrame with columns ['file_name', 'model', 'MLAE']

    Returns:
        pd.DataFrame: Summary table showing the best model for each image type
                      and their average MLAE.
    """
    # Extract the image type from the file name (aliased, antialiased, vectorized)
    df['image_type'] = df['file_name'].str.extract('(aliased|antialiased|vectorized)')

    # Calculate the average MLAE for each image type and model
    avg_mlae = df.groupby(['image_type', 'model'])['MLAE'].mean().reset_index()

    # Identify the best model for each image type (lowest MLAE)
    best_model = avg_mlae.loc[avg_mlae.groupby('image_type')['MLAE'].idxmin()]

    # Rename columns for better readability
    best_model = best_model.rename(columns={
        'model': 'Best Model',
        'MLAE': 'Average MLAE'
    })

    # Sort by image type for readability
    best_model = best_model.sort_values(by='image_type').reset_index(drop=True)

    return best_model

def analyze_image_type_per_model(df):
    """
    Analyze and determine which image type works best for each model based on average MLAE.

    Parameters:
        df (pd.DataFrame): DataFrame with columns ['file_name', 'model', 'MLAE']

    Returns:
        pd.DataFrame: Summary table showing the best image type for each model
                      and their average MLAE for that image type.
    """
    # Extract the image type from the file name
    df['image_type'] = df['file_name'].str.extract(r'(aliased|antialiased|vectorized)')
    
    # Group by model and image type to calculate average MLAE
    avg_mlae_per_model = df.groupby(['model', 'image_type'])['MLAE'].mean().reset_index()
    
    # Find the best image type (lowest MLAE) for each model
    best_image_type_per_model = avg_mlae_per_model.loc[
        avg_mlae_per_model.groupby('model')['MLAE'].idxmin()
    ]
    
    # Sort the results for clarity
    best_image_type_per_model = best_image_type_per_model.sort_values(by='MLAE').reset_index(drop=True)
    
    return best_image_type_per_model

import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import fitz  # PyMuPDF

def plot_mlae_per_image(df, image_folder, model_colors, xlim=(2.5, 4.5), reference_line=3):
    """
    Plots average MLAE per image type for different models with corresponding images.
    
    Parameters:
    - df (DataFrame): DataFrame containing 'image_type', 'model', 'MLAE', and 'file_name'.
    - image_folder (str): Path to the folder containing images.
    - model_colors (dict): Dictionary mapping model names to their colors.
    - xlim (tuple): Tuple specifying x-axis limits.
    - reference_line (float): x-coordinate for the vertical reference line.
    """
    # Extract image types and calculate average MLAE
    df["image_type"] = df["file_name"].apply(lambda x: "_".join(x.split("_")[:-1]))
    average_mlae = df.groupby(["image_type", "model"])["MLAE"].mean().reset_index()
    image_types = average_mlae["image_type"].unique()

    # Create grid layout
    nrows = len(image_types)
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, nrows * 2), gridspec_kw={"width_ratios": [1, 4]})

    # Function to load and display images
    def load_and_display_image(file_path, image_type):
        if file_path.endswith(".png") and os.path.exists(file_path):
            img = Image.open(file_path)
            if "black" in image_type:
                img = img.convert("L")
            else:
                img = img.convert("RGB")
        elif file_path.endswith(".pdf") and os.path.exists(file_path):
            pdf_document = fitz.open(file_path)
            page = pdf_document[0]
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pdf_document.close()
        else:
            raise FileNotFoundError(f"File not found or unsupported format: {file_path}")
        return img

    # Iterate over image types
    for i, image_type in enumerate(image_types):
        subset = average_mlae[average_mlae["image_type"] == image_type]

        # Left column: Image
        ax_img = axes[i, 0] if nrows > 1 else axes[0]
        file_name = df[df["image_type"] == image_type]["file_name"].iloc[0]
        file_path = os.path.join(image_folder, file_name)
        try:
            img = load_and_display_image(file_path, image_type)
            ax_img.imshow(img, cmap="gray" if "black" in image_type else None)
            ax_img.axis("off")
            ax_img.text(
                0.5, -0.2, image_type, transform=ax_img.transAxes, ha="center", va="top", fontsize=10, color="black"
            )
        except FileNotFoundError:
            ax_img.text(0.5, 0.5, "Image not found", ha="center", va="center", fontsize=12, color="red")
            ax_img.axis("off")

        # Right column: Scatter plot
        ax_plot = axes[i, 1] if nrows > 1 else axes[1]
        for model, color in model_colors.items():
            model_data = subset[subset["model"] == model]
            sizes = model_data["MLAE"].apply(lambda x: 300 if x < 3 else 100)
            ax_plot.scatter(
                model_data["MLAE"],
                [i] * len(model_data),
                color=color,
                s=sizes,
                alpha=0.8,
            )
        
        # Add a vertical reference line
        ax_plot.axvline(x=reference_line, color='red', linestyle='--', linewidth=1, label="Reference (MLAE = 3)")

        # Hide all spines except the bottom one
        ax_plot.spines["top"].set_visible(False)
        ax_plot.spines["right"].set_visible(False)
        ax_plot.spines["left"].set_visible(False)

        # Set x-axis limits
        ax_plot.set_xlim(*xlim)

        # Handle bottom spine and x-tick labels
        if i == len(image_types) - 1:
            ax_plot.spines["bottom"].set_visible(True)
            ax_plot.tick_params(axis="x", which="both", bottom=True, labelbottom=True)
        else:
            ax_plot.spines["bottom"].set_visible(False)
            ax_plot.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        # Hide y-ticks and labels
        ax_plot.tick_params(axis="y", left=False, labelleft=False)
        ax_plot.grid(axis="x", linestyle="--", alpha=0.8)

    # Add a title
    fig.suptitle("Average MLAE for Each Image Per Model", fontsize=16, weight="bold")

    # Add legend
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=model, markersize=10, markerfacecolor=color)
        for model, color in model_colors.items()
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(1.1, 0.97),
        frameon=False,
        title="Models",
        fontsize=12
    )

    # Adjust layout
    plt.tight_layout()
    plt.show()

