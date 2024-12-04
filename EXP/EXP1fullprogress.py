# Standard library imports
import os
import sys
import time
import json
import random
import uuid
from collections import Counter
from contextlib import redirect_stdout

# Third-party library imports
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from transformers import AutoProcessor, AutoModelForVision2Seq, get_scheduler, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import login

# Custom modules
sys.path.append("../")  # Add custom module path
import LLMP as L

# Hugging Face login using the token
login('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')

""" Define Global Numbers """
IMAGES_PER_TASK = 10
MAX_EPOCHS = 1
VAL_CHECK_INTERVAL = 90
CHECKPOINT_DIR = "/hpcstor6/scratch01/h/huuthanhvy.nguyen001/cache/checkpoint"
last_checkpoint = os.path.join(CHECKPOINT_DIR, "last.ckpt")  # Use the correct checkpoint directory

# Constants
CSV_FILENAME = "EXP1_results.csv"
EVALUATION_TIME = 3

# Task-specific queries
QUERIES = {
    "position_common_scale": "Estimate the block's vertical position (range: 0-60, top to bottom). Number only. No explanation.",
    "position_non_aligned_scale": "Estimate the block's vertical position (range: 0-60, top to bottom). Number only. No explanation.",
    "length": "Estimate the line length from top to bottom (range: 0-100). Number only. No explanation.",
    "direction": "Estimate the line's direction (range: 0-359 degrees). Number only. No explanation.",
    "angle": "Estimate the angle (range: 0-90 degrees). Number only. No explanation.",
    "area": "Estimate the area of a circle, ensuring your answer falls within the range of 3.14 to 5026.55 square units. Assume the circle fits within a 100x100 pixel image. Provide only the numeric value, no explanation.",
    "volume": "Estimate the volume of a cube, with your answer restricted to the range of 1 to 8000 cubic units. Assume the cube fits within a 100x100 pixel image. Provide only the numeric value, no explanation.",
    "curvature": "Estimate the line curvature (range: 0.000 to 0.088) of a Bezier curve constrained within a 100x100 pixel space. Provide only the numeric curvature value (up to 3 decimal places), no explanation.",
    "shading": "Estimate shading density (range: 0-100). Number only. No explanation."
}


""" Step 1: Generate EXP1 DATASET """

def generate_dataset_exp1(main_output_dir, images_per_task=IMAGES_PER_TASK):
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

def display_training_samples_exp1(main_output_dir="finetuning-EXP1-test", num_images=2):

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

def verify_dataset(main_output_dir="finetuning-EXP1-test"):
    """
    Verify the dataset by checking the number of images for each task,
    ensuring uniqueness across datasets, and identifying overlaps.

    Parameters:
        main_output_dir (str): Directory where datasets and images are saved.

    Returns:
        None
    """
    

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


""" Step 2: Finetune EXP1 DATASET """

def configure_training_params():

    """Set training parameters."""
    return {

        "model_id": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "batch_size": 1,
        "learning_rate": 0.0001,
        "weight_decay": 0.01,
        "max_epochs": MAX_EPOCHS,
        "gradient_accumulation": 8,
        "log_interval": 100,
        "val_check_interval": VAL_CHECK_INTERVAL,
    }

def setup_directories(base_dir):

    """Set up directories for fine-tuning."""
    directories = {
        "base_dir": base_dir,
        "data_dir": os.path.join(base_dir, 'json'),
        "image_folder": os.path.join(base_dir, 'images'),
        "save_dir": os.path.join(base_dir, 'fine_tuned_model'),
        "log_dir": base_dir
    }
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    return directories

def initialize_processor(model_id):
    """Initialize the processor for the model."""
    return AutoProcessor.from_pretrained(model_id)

def define_data_module(data_dir, image_folder, processor, batch_size):

    """Define the data module for PyTorch Lightning."""
    class ImageTextDataModule(LightningDataModule):

        def __init__(self):
            super().__init__()
            self.data_dir = data_dir
            self.image_folder = image_folder
            self.processor = processor
            self.batch_size = batch_size
            self.train_path = os.path.join(self.data_dir, 'train_dataset.json')
            self.val_path = os.path.join(self.data_dir, 'val_dataset.json')

        def prepare_data(self):
            """Prepare data by verifying the existence of train and validation datasets."""

            # Check if the required files exist
            if not os.path.exists(self.train_path) or not os.path.exists(self.val_path):
                raise FileNotFoundError("Train or validation dataset not found. Please ensure both files exist.")

            print(f"Train dataset located at: {self.train_path}")
            print(f"Validation dataset located at: {self.val_path}")

            print("Data preparation complete.")

        def setup(self, stage=None):

            self.train_data = Dataset.from_json(self.train_path)
            self.val_data = Dataset.from_json(self.val_path)

            print("Training and validation datasets successfully loaded.")


        def process(self, examples):
            texts = [
                f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n<|image|> {item['question']} <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{item['value']}<|eot_id|>"
                for item in examples
            ]
            images = []
            for item in examples:
                image_path = os.path.join(self.image_folder, os.path.basename(item["image"]))
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image not found at: {image_path}")
                images.append(Image.open(image_path).convert("RGB"))

            batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
            labels = batch["input_ids"].clone()
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            labels[labels == 128256] = -100  # Mask image token index for images
            batch["labels"] = labels
            return batch


        def collate_fn(self, batch):
            return self.process(batch)

        def train_dataloader(self):
            return DataLoader(self.train_data, batch_size=self.batch_size, collate_fn=self.collate_fn)

        def val_dataloader(self):
            return DataLoader(self.val_data, batch_size=self.batch_size, collate_fn=self.collate_fn)

    return ImageTextDataModule()

def define_model(model_id, learning_rate, weight_decay):
    """Define the model for fine-tuning."""
    class VisionTextModel(LightningModule):
        def __init__(self, model_id, learning_rate, weight_decay):
            super().__init__()
            # Save hyperparameters
            self.save_hyperparameters()

            # Configure model
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

            self.model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                quantization_config=bnb_config,
                low_cpu_mem_usage=True,
                cache_dir = "/hpcstor6/scratch01/h/huuthanhvy.nguyen001/cache"
            )

            # Configure LoRA
            peft_config = LoraConfig(
                lora_alpha=64,
                lora_dropout=0.1,
                r=128,
                bias="none",
                target_modules=["q_proj", "v_proj"],
                task_type="FEATURE_EXTRACTION",
            )

            self.model = get_peft_model(self.model, peft_config)
            self.model.tie_weights()

        def save_pretrained(self, save_directory):
            """Save the underlying Hugging Face model."""
            os.makedirs(save_directory, exist_ok=True)
            self.model.save_pretrained(save_directory)

        def forward(self, input_ids=None, attention_mask=None, pixel_values=None, labels=None, **kwargs):
            """Forward pass for the model."""
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
                **kwargs
            )

        def training_step(self, batch, batch_idx):
            """Training step."""
            outputs = self(**batch)
            loss = outputs.loss
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return loss

        def validation_step(self, batch, batch_idx):
            """Validation step."""
            outputs = self(**batch)
            loss = outputs.loss
            self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return loss

        def configure_optimizers(self):
            """Configure optimizers and schedulers."""
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,  # Access learning rate from saved hyperparameters
                weight_decay=self.hparams.weight_decay  # Access weight decay from saved hyperparameters
            )
            scheduler = get_scheduler(
                name="linear",
                optimizer=optimizer,
                num_warmup_steps=100,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    # Return an instance of the model with the provided hyperparameters
    return VisionTextModel(
        model_id=model_id,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )

def fine_tune_exp1(base_dir):
    """Main function to fine-tune the model."""
    dirs = setup_directories(base_dir)
    params = configure_training_params()

    # Initialize components
    processor = initialize_processor(params["model_id"])
    data_module = define_data_module(dirs["data_dir"], dirs["image_folder"], processor, params["batch_size"])
    model = define_model(params["model_id"], params["learning_rate"], params["weight_decay"])

    # Loggers
    tensorboard_logger = TensorBoardLogger(save_dir=dirs["base_dir"], name="finetune_logs")
    csv_logger = CSVLogger(save_dir=dirs["base_dir"], name="metrics_logs")

    # Define checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(CHECKPOINT_DIR, "checkpoints"),
        filename="finetuned-{epoch:02d}",
        save_last=True,
        save_top_k=-1,
    )

    # Universal checkpoint restoration utility
    def restore_checkpoint(model, checkpoint_dir):
        """Restore checkpoint if available, filtering incompatible keys."""
        last_ckpt_path = os.path.join(checkpoint_dir, "last.ckpt")

        if os.path.exists(last_ckpt_path):
            print(f"Resuming from checkpoint: {last_ckpt_path}")

            checkpoint = torch.load(last_ckpt_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)  # Handle Lightning or PyTorch states
            filtered_state_dict = {k: v for k, v in state_dict.items() if "quant" not in k and "absmax" not in k}
            
            try:
                model.load_state_dict(filtered_state_dict, strict=False)
            except Exception as e:
                print(f"Failed to load checkpoint: {e}. Starting from scratch.")

        else:
            print("No checkpoint found. Starting from scratch.")

    # Attempt to restore the checkpoint
    restore_checkpoint(model, CHECKPOINT_DIR)

    # Trainer definition
    trainer = pl.Trainer(
        max_epochs=params["max_epochs"],
        val_check_interval=params["val_check_interval"],
        logger=[tensorboard_logger, csv_logger],
        callbacks=[checkpoint_callback],
        accelerator="gpu",
        devices=1,
        default_root_dir=dirs["base_dir"],
    )

    # Start fine-tuning
    print("Starting fine-tuning...")
    trainer.fit(model, datamodule=data_module)

    # Save the model
    os.makedirs(dirs["save_dir"], exist_ok=True)
    try:
        model.save_pretrained(dirs["save_dir"])
        print(f"Model saved to {dirs['save_dir']}")
    except AttributeError:
        # Fallback to saving state_dict
        torch.save(model.state_dict(), os.path.join(dirs["save_dir"], "model_state_dict.pth"))
        print(f"State_dict saved to {dirs['save_dir']} as fallback.")


""" Step 3: Evaluate EXP1 """

def load_test_dataset(test_dataset_path):
    """
    Load the test dataset from JSON.
    """
    if not os.path.exists(test_dataset_path):
        raise FileNotFoundError(f"Test dataset not found at: {test_dataset_path}")
    return pd.read_json(test_dataset_path)

def ensure_dir_exists(directory):
    """
    Ensure that a directory exists.
    """
    os.makedirs(directory, exist_ok=True)

def save_results_to_csv(results, results_dir, csv_filename):
    """
    Save results to a CSV file.
    """
    ensure_dir_exists(results_dir)
    file_path = os.path.join(results_dir, csv_filename)
    results_df = pd.DataFrame(results)
    results_df.to_csv(file_path, index=False)
    print(f"Results saved to: {file_path}")

def Runexp1(test_dataset_path, results_dir, model_dir, csv_filename, image_base_dir):
    """
    Main function to run all tasks, evaluate models, and save results to CSV.
    """
    # Load the test dataset
    test_data = load_test_dataset(test_dataset_path)

    # Ensure the results directory exists
    ensure_dir_exists(results_dir)

    # Load environment variables and authenticate with Hugging Face
    load_dotenv()

    # Initialize models
    models = {
        "CustomLLaMA": L.llamafinetuned(model_dir)  # Use the fine-tuned model
    }


    # Initialize results storage
    results = []

    # Process each entry in the test dataset
    for index, row in test_data.iterrows():
        task_name = row["task"]
        query = QUERIES.get(task_name, "No query available for this task.")  # Retrieve the query for the task

        # Fix image path by avoiding duplicate "images/" in the path
        image_relative_path = row["image"]
        if image_relative_path.startswith("images/"):
            image_path = os.path.join(image_base_dir, image_relative_path[len("images/"):])
        else:
            image_path = os.path.join(image_base_dir, image_relative_path)

        ground_truth = row["value"]

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        # Load the image
        grayscale_image = np.array(Image.open(image_path).convert("L"))  # Convert to grayscale

        # Evaluate each model EVALUATION_TIME times
        for run in range(EVALUATION_TIME):
            print(f"Run {run + 1} for task: {task_name}, image: {image_path}")

            for model_name, model_instance in models.items():
                torch.cuda.empty_cache()
                print(f"Running {model_name} on {image_path}...")

                start_time = time.time()
                raw_answer = model_instance.query(query, grayscale_image)
                elapsed_time_ms = (time.time() - start_time) * 1000

                # Save the result
                results.append({
                    "task_name": task_name,
                    "query": query,  # Include the query in the results
                    "run": run + 1,
                    "model_name": model_name,
                    "image_path": image_path,
                    "ground_truth": ground_truth,
                    "raw_answer": raw_answer,
                    "time_ms": elapsed_time_ms
                })

    # Save all results to CSV
    save_results_to_csv(results, results_dir, csv_filename)
    print("All tasks completed successfully.")

def setup_logging(output_dir, experiment_name):
    """
    Set up the logging file for the experiment.
    """
    log_file = os.path.join(output_dir, f"{experiment_name}_log.txt")
    return log_file

def run_multiple_experimentsEXP1(num_experiments=3):
    """
    Run multiple experiments sequentially, generating datasets, fine-tuning,
    and running evaluations for each experiment.
    """
    for exp_num in range(1, num_experiments + 1):
        exp_name = f"EXP{exp_num}"
        main_output_dir = f"./finetuning-{exp_name}-test/"

        # Ensure the output directory exists
        os.makedirs(main_output_dir, exist_ok=True)

        log_file = setup_logging(main_output_dir, exp_name)
        image_base_dir = os.path.join(main_output_dir, "images")

        with open(log_file, "w") as f, redirect_stdout(f):
            try:
                print(f"\n{'='*20} STARTING {exp_name} {'='*20}")

                timesleep = 10  # Time delay between steps
                start_time = time.time()

                # Step 1: Generate dataset
                print(f"\n{'='*20} DATASET GENERATION {'='*20}")
                generate_dataset_exp1(main_output_dir=main_output_dir)
                dataset_time = time.time()
                print(f"Dataset generation completed in {dataset_time - start_time:.2f} seconds.")
                time.sleep(timesleep)

                # Step 2: Fine-tune dataset
                print(f"\n{'='*20} FINE-TUNING {'='*20}")
                fine_tune_exp1(base_dir=main_output_dir)
                fine_tune_time = time.time()
                print(f"Fine-tuning completed in {fine_tune_time - dataset_time:.2f} seconds.")
                time.sleep(timesleep)

                # Step 3: Run experiment
                print(f"\n{'='*20} RUNNING EXPERIMENT {'='*20}")
                test_dataset_path = os.path.join(main_output_dir, "json/test_dataset.json")
                results_dir = os.path.join(main_output_dir, "EXP-Results")
                model_dir = os.path.join(main_output_dir, "fine_tuned_model")
                csv_filename = f"{exp_name}_results.csv"

                # Ensure required directories exist
                ensure_dir_exists(results_dir)
                ensure_dir_exists(image_base_dir)

                Runexp1(
                    test_dataset_path=test_dataset_path,
                    results_dir=results_dir,
                    model_dir=model_dir,
                    csv_filename=csv_filename,
                    image_base_dir=image_base_dir
                )
                experiment_time = time.time()

                print(f"\n{'='*20} {exp_name} SUMMARY {'='*20}")
                print(f"Dataset Generation: {dataset_time - start_time:.2f} seconds")
                print(f"Fine-tuning: {fine_tune_time - dataset_time:.2f} seconds")
                print(f"Experiment Running: {experiment_time - fine_tune_time:.2f} seconds")
                print(f"Total Time: {experiment_time - start_time:.2f} seconds")
                print(f"{exp_name} COMPLETED SUCCESSFULLY.")

            except Exception as e:
                print(f"\nERROR: An error occurred during {exp_name} execution:")
                print(f"Error details: {str(e)}")
                raise