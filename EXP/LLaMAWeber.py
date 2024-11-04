import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import json
from datasets import Dataset
from PIL import Image
import os
from transformers import AutoModelForVision2Seq, AutoProcessor, Trainer, TrainingArguments, TrainerCallback
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
import torch
from PIL import Image
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
import torch


# Change the directory to where the JSON file is located
os.chdir('./outputWEBER-5000/json')

# Load the JSON file into a pandas DataFrame
df = pd.read_json('combined_dataset.json')

# Split dataset into train+val (80%) and test (20%)
train_val, test = train_test_split(df, test_size=0.2, random_state=42)

# Split train+val into train (80% of 80% = 64% of total) and val (20% of 80% = 16% of total)
train, val = train_test_split(train_val, test_size=0.2, random_state=42)

# Save the train, val, and test datasets as separate JSON files
train.to_json('train_dataset.json', orient='records', indent=4)
val.to_json('val_dataset.json', orient='records', indent=4)
test.to_json('test_dataset.json', orient='records', indent=4)

print(f"Train size: {len(train)}, Validation size: {len(val)}, Test size: {len(test)}")


# Create a list of dictionaries from the DataFrame
train_dataset = [
    {
        'id': row['id'],          # Extract the 'id' column
        'image': row['image'],    # Extract the 'image' column
        'question': row ['question'],  # Add the question for validation
        'value': row['value']     # Extract the 'value' column as the 'value'
    }
    for _, row in train.iterrows()  # Iterate over DataFrame rows
]

# Convert the list of dictionaries into a Hugging Face Dataset
train_data = Dataset.from_list(train_dataset)

# Print the dataset structure for verification
print(train_data)

# Assuming 'val' is your pandas DataFrame for validation data

# Create a list of dictionaries from the DataFrame
validation_dataset = [
    {
        'id': row['id'],          # Extract the 'id' column
        'image': row['image'],    # Extract the 'image' column
        'question': row ['question'],  # Add the question for validation
        'value': row['value']     # Extract the 'value' column as the 'value'
    }
    for _, row in val.iterrows()  # Iterate over DataFrame rows
]

# Convert the list of dictionaries into a Hugging Face Dataset
validation_data = Dataset.from_list(validation_dataset)

# Print the dataset structure for verification
print(validation_data)

# Assuming 'test' is your pandas DataFrame for test data

# Create a list of dictionaries from the DataFrame
test_dataset = [
    {
        'id': row['id'],          # Extract the 'id' column
        'image': row['image'],    # Extract the 'image' column
        'question': row ['question'],  # Add the question for validation
        'value': row['value']     # Extract the 'value' column as the 'value'
    }
    for _, row in test.iterrows()  # Iterate over DataFrame rows
]

# Convert the list of dictionaries into a Hugging Face Dataset
test_data = Dataset.from_list(test_dataset)

# Print the dataset structure for verification
print(test_data)


# Define the model ID and login
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
login('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')

# Load Bits and Bytes Configuration for quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_use_double_quant=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load the model with quantization
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
)

# Load the processor
processor = AutoProcessor.from_pretrained(model_id)

# Define LoRA config based on QLoRA experiments
peft_config = LoraConfig(
    lora_alpha=64,
    lora_dropout=0.1,
    r=64,
    bias="none",
    target_modules=["q_proj", "v_proj"],  # LoRA targets these transformer layers
    task_type="FEATURE_EXTRACTION",  # Task type for Feature_extration
)

# Apply LoRA adapters to the loaded model
model = get_peft_model(model, peft_config)



# Path to the folder containing the images
image_folder = "/home/huuthanhvy.nguyen001/LLMP/EXP/outputWEBER-5000/images"

# Function to process the examples
def process(examples):
    # Construct the prompt asking for the angle in the image
    texts = [
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n<|image|> {item['question']} <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{item['value']}<|eot_id|>"
        for item in examples
    ]
    
    # Load images from the folder
    images = [
        Image.open(os.path.join(image_folder, item["image"])).convert("RGB")
        for item in examples
    ]

    # Assuming `processor` is defined elsewhere in the code
    # The processor will handle the tokenization of the text and processing of the image
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    
    # Clone the input IDs to create labels, masking padding and image token index
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == 128256] = -100  # Mask image token index for images

    # Assign the correct output (angle value) as the label for each example
    batch["labels"] = labels

    # Move the batch to bfloat16 and to the GPU (cuda) for faster training
    batch = batch.to(torch.bfloat16).to("cuda")
    
    return batch

from PIL import Image
import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
from transformers import Trainer, TrainingArguments, TrainerCallback, TrainerState, TrainerControl
import numpy as np
from sklearn.metrics import mean_absolute_error
from huggingface_hub import login

# Define the model ID and login
login('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')

# Dynamically set output_dir based on the number of images
name = "finetuned-5000-images-5-epochs-weber" # Change this number as needed
output_dir = f"generated_images_{name}"  # E.g., "generated_images_450"
log_file_path = f"training_logs_{name}.txt"  # Dynamic log file path

# Modify TrainingArguments to include evaluation strategy
training_args = TrainingArguments(
    output_dir=output_dir,
    push_to_hub=False,
    num_train_epochs=5,
    logging_steps=100,
    evaluation_strategy="epoch",
    remove_unused_columns=False,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    learning_rate=0.0001,
    weight_decay=0.01,
    adam_beta2=0.999,
    max_grad_norm=1.0,
    save_strategy="no",
    optim="adamw_hf",
    save_total_limit=1,
    bf16=True,
    dataloader_pin_memory=False,
)

# Custom callback to log both training and validation metrics
class LogMetricsCallback(TrainerCallback):
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.training_logs = []
        self.validation_logs = []

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        # Log training loss if available
        if "loss" in logs:
            with open(self.log_file_path, "a") as f:
                f.write(f"Training loss: {logs['loss']} at step {state.global_step}\n")
                print(f"Training loss: {logs['loss']} at step {state.global_step}")
            self.training_logs.append((state.global_step, logs["loss"]))
        
        # Log validation loss if available
        if "eval_loss" in logs:
            with open(self.log_file_path, "a") as f:
                f.write(f"Validation loss: {logs['eval_loss']} at step {state.global_step}\n")
                print(f"Validation loss: {logs['eval_loss']} at step {state.global_step}")
            self.validation_logs.append((state.global_step, logs["eval_loss"]))

log_metrics_callback = LogMetricsCallback(log_file_path=log_file_path)

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Assuming `model` is loaded before this section
model.tie_weights()  # Tie the weights after loading the model

# Trainer setup including validation data
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=process,
    train_dataset=train_data,
    eval_dataset=validation_data,
    callbacks=[log_metrics_callback],
)

# Train the model with validation
trainer.train()

model.save_pretrained("my_finetuned_llama_all_images_weber_5000images_5epoch")

print("All set finetuning - Yeah!")