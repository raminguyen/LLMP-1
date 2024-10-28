import pandas as pd
from sklearn.model_selection import train_test_split
import json
from datasets import Dataset
from PIL import Image
import os
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
import torch
import matplotlib.pyplot as plt
from transformers import Trainer, TrainingArguments, TrainerCallback, TrainerState, TrainerControl
import numpy as np
from sklearn.metrics import mean_absolute_error

# Path to the folder containing the images
image_folder = "/home/huuthanhvy.nguyen001/LLMP/EXP/output/images"

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
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    
    # Clone the input IDs to create labels, masking padding and image token index
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == 128256] = -100  # Mask image token index for images

    batch["labels"] = labels
    batch = batch.to(torch.bfloat16).to("cuda")
    
    return batch

# Change the directory to where the JSON file is located
os.chdir('./output/json')

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
        'id': row['id'],
        'image': row['image'],
        'question': row['question'],
        'value': row['value']
    }
    for _, row in train.iterrows()
]
train_data = Dataset.from_list(train_dataset)
print(train_data)

validation_dataset = [
    {
        'id': row['id'],
        'image': row['image'],
        'question': row['question'],
        'value': row['value']
    }
    for _, row in val.iterrows()
]
validation_data = Dataset.from_list(validation_dataset)
print(validation_data)

test_dataset = [
    {
        'id': row['id'],
        'image': row['image'],
        'question': row['question'],
        'value': row['value']
    }
    for _, row in test.iterrows()
]
test_data = Dataset.from_list(test_dataset)
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
    target_modules=["q_proj", "v_proj"],
    task_type="FEATURE_EXTRACTION",
)

# Apply LoRA adapters to the loaded model
model = get_peft_model(model, peft_config)

# Dynamically set output_dir based on the number of images
name = "test"
output_dir = f"generated_images_{name}"
log_file_path = f"training_logs_{name}.txt"

# Modify TrainingArguments to include evaluation strategy
training_args = TrainingArguments(
    output_dir=output_dir,
    push_to_hub=False,
    num_train_epochs=3,
    logging_steps=100,
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
        if logs is not None and "eval_loss" in logs:
            with open(self.log_file_path, "a") as f:
                f.write(f"Validation loss: {logs['eval_loss']} at step {state.global_step}\n")
                print(f"Validation loss: {logs['eval_loss']} at step {state.global_step}")
            self.validation_logs.append((state.global_step, logs["eval_loss"]))

log_metrics_callback = LogMetricsCallback(log_file_path=log_file_path)

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)
model.tie_weights()

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

model.save_pretrained("finetuned_llama_EXP1")
