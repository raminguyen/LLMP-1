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
    "CustomLLaMA": L.llamafinetuned("./outputEXP3-5000/json/finetuned_llama_EXP3"),
    "LLaMA": L.llama("meta-llama/Llama-3.2-11B-Vision-Instruct"),
    "GeminiProVision": L.GeminiProVision(),
    "Gemini1_5Flash": L.Gemini1_5Flash()
}


# Set up the number of images and sleep duration
num_images = 20
sleep_duration = 10  # Sleep for 10 seconds between each task

# Create an instance of the Runexp3 class
experiment = L.Runexp3()

"""

# Run experiment for "type1" task
print("Starting task: type1")
experiment.Runexp3(num_images=num_images, model_instances=model_instances, tasks="type1")
time.sleep(sleep_duration)

# Run experiment for "type2" task
print("Starting task: type2")
experiment.Runexp3(num_images=num_images, model_instances=model_instances, tasks="type2")
time.sleep(sleep_duration)

# Run experiment for "type3" task
print("Starting task: type3")
experiment.Runexp3(num_images=num_images, model_instances=model_instances, tasks="type3")
time.sleep(sleep_duration)

# Run experiment for "type4" task
print("Starting task: type4")
experiment.Runexp3(num_images=num_images, model_instances=model_instances, tasks="type4")
time.sleep(sleep_duration)

"""

# Run experiment for "type5" task
print("Starting task: type5")
experiment.Runexp3(num_images=num_images, model_instances=model_instances, tasks="type5")
time.sleep(sleep_duration)

print("All tasks completed. Yeah!")
