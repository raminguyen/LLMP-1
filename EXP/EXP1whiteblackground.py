import os
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


def display_all_images(folder, canvas_size=100):
    """Display all vectorized, aliased, and anti-aliased images."""
    variants = ["white", "black"]
    image_types = ["aliased", "antialiased", "vectorized"]

    fig, axes = plt.subplots(len(variants), len(image_types), figsize=(15, 8))
    fig.subplots_adjust(left=0.08, right=0.92, top=0.9, bottom=0.1, wspace=0.2, hspace=0.2)

    for i, variant in enumerate(variants):
        for j, image_type in enumerate(image_types):
            ax = axes[i, j]
            try:
                if image_type == "vectorized":
                    # PDF Handling
                    file = os.path.join(folder, f"{image_type}_image_{variant}.pdf")
                    pdf_document = fitz.open(file)
                    page = pdf_document[0]  # Render the first page
                    pix = page.get_pixmap(dpi=300)  # High-resolution rendering
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ax.imshow(img)
                    pdf_document.close()
                else:
                    # PNG Handling
                    file = os.path.join(folder, f"{image_type}_image_{variant}.png")
                    img = Image.open(file)
                    ax.imshow(img, cmap="gray")
                ax.axis("off")
            except FileNotFoundError:
                ax.text(0.5, 0.5, "File Not Found", fontsize=12, ha='center', va='center')
                ax.axis("off")

        fig.text(
            0.05, 0.75 - i * 0.5,
            f"{variant.capitalize()} Background",
            fontsize=14, fontweight="bold", color="black" if variant == "white" else "white",
            ha="center", va="center",
            bbox=dict(facecolor=variant, edgecolor="none", boxstyle="round,pad=0.3"),
            rotation=90
        )

    fig.suptitle("Image Comparison: Aliased, Anti-Aliased, and Vectorized", fontsize=16, fontweight="bold")
    plt.show()


def clean_experiment_data(file_path):
    """
    Cleans and processes experiment results from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file containing experiment results.

    Returns:
        pd.DataFrame: A cleaned DataFrame with extracted and formatted data.
    """
    # Set pandas display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    # Read the CSV file
    df = pd.read_csv(file_path, header=1, names=['Model', 'Image Path', 'Answer'])

    # Clean the Answer column: remove newline characters and unnecessary text
    df['Answer'] = df['Answer'].astype(str).str.strip()  # Strip leading/trailing whitespace
    df['Answer'] = df['Answer'].str.extract(r'(\d+)$').astype(float)  # Extract numeric answers

    # Extract the last part of the image path and add it as a new column
    df['Image Name'] = df['Image Path'].str.split('/').str[-1]

    # Drop the 'Image Path' column
    df = df.drop('Image Path', axis=1)

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
   # Remove NaN values
    valid_mask = ~np.isnan(answers)
    gt_filtered = np.array(gt)[valid_mask]
    answers_filtered = answers[valid_mask]

    if len(answers_filtered) == 0:
        return np.nan

    return np.log2(mean_absolute_error(gt_filtered, answers_filtered) + 0.125)

# Function to calculate MLAE for a single row
def calculate_row_mlae(row):
    # Ground truth value
    ground_truth = 20

    """
    Calculate MLAE for a single row using its Answer and the ground truth.

    Parameters:
        row (pd.Series): A single row of the DataFrame.

    Returns:
        float: The MLAE for the row.
    """
    answer = np.array([row['Answer']])
    gt = np.array([ground_truth]) 
    return calculate_mlae(gt, answer)  


def plot_mlae_heatmap(df):
    """
    Plot a heatmap showing MLAE values for each model and image.

    Parameters:
        df (pd.DataFrame): DataFrame containing MLAE values with columns 
                           'Image Name', 'Model', and 'MLAE'.

    Returns:
        None
    """
    # Pivot the DataFrame for heatmap data
    heatmap_data = df.pivot(index='Image Name', columns='Model', values='MLAE')

    # Import libraries
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Plot the heatmap
    plt.figure(figsize=(12, 8))

    sns.heatmap(heatmap_data, 
                annot=True, 
                cmap='coolwarm', 
                cbar_kws = {'label': "MLAE"},
                annot_kws = {"fontsize": 12})

    plt.title('MLAE Heatmap for Each Model and Images')

    plt.xlabel('Model Name', fontsize = 14 )

    plt.ylabel('Image Name', fontsize = 14 )

    plt.xticks (fontsize=12)

    plt.yticks (fontsize=14)

    plt.show()