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
    "CustomLLaMA": L.llamafinetuned("./EXPs-5000-10epochs/EXP5-Results/finetuning-EXP5-5000-10epochs/fine_tuned_model"),
    "LLaMA": L.llama("meta-llama/Llama-3.2-11B-Vision-Instruct"),
    "GeminiProVision": L.GeminiProVision(),
    "Gemini1_5Flash":L.Gemini1_5Flash()
}

# Set up the number of images and sleep time
num_images = 55
timesheet = 10 # Sleep for 5 seconds between each task

# Create an instance of the Runexp1 class
experiment = L.Runexp5()

# Run experiments for each task with sleep intervals between them
print("Starting task: 10")
experiment.Runexp5(num_images=num_images, model_instances=model_instances, tasks='10')
time.sleep(timesheet)

print("Starting task: 100")
experiment.Runexp5(num_images=num_images, model_instances=model_instances, tasks='100')
time.sleep(timesheet)

print("Starting task: 1000")
experiment.Runexp5(num_images=num_images, model_instances=model_instances, tasks='1000')
time.sleep(timesheet)

