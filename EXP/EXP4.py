import sys
import os
import torch
from dotenv import load_dotenv
import time
from huggingface_hub import login

# Add LLMP path
sys.path.append('../')
import LLMP as L

# Load environment variables
load_dotenv()

# Clear GPU cache
torch.cuda.empty_cache()

# Hugging Face login using the token
login('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')


# Set up model instances
model_instances = {
    "gpt4o": L.GPTModel("gpt-4o"),
    "CustomLLaMA": L.llamafinetuned("./EXPs-5000-10epoch-lora/finetuning-EXP4-5000-10epochs-lora/fine_tuned_model"), 
    "LLaMA": L.llama("meta-llama/Llama-3.2-11B-Vision-Instruct"),
    "GeminiProVision": L.GeminiProVision(),
    "Gemini1_5Flash": L.Gemini1_5Flash()
}


# Set up the number of images and sleep duration
num_images = 55
sleep_duration = 10  # Sleep for 10 seconds between each task

# Create an instance of the Runexp3 class
experiment = L.Runexp4()

# Run experiment for "type1" task
print("Starting task: framed")
experiment.Runexp4(num_images=num_images, model_instances=model_instances, tasks="framed")
time.sleep(sleep_duration)

# Run experiment for "type2" task
print("Starting task: unframed")
experiment.Runexp4(num_images=num_images, model_instances=model_instances, tasks="unframed")
time.sleep(sleep_duration)

print("All tasks completed. Yeah!")
