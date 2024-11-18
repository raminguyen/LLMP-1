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

# Set up the number of images and sleep time
num_images = 55
timesheet = 10 # Sleep for 5 seconds between each task

# Set up model instances 
model_instances = {
    "gpt4o": L.GPTModel("gpt-4o"), 
    "CustomLLaMA": L.llamafinetuned("./EXPs-5000-10epochs/EXP1-Results/finetuning-EXP1-5000-10epochs/fine_tuned_model"),
    "LLaMA": L.llama("meta-llama/Llama-3.2-11B-Vision-Instruct"),
    "GeminiProVision": L.GeminiProVision(),
    "Gemini1_5Flash": L.Gemini1_5Flash()
}

# Create an instance of the Runexp1 class
experiment = L.Runexp1()

print("Starting task: area")
experiment.Runexp1(num_images=num_images, model_instances=model_instances, tasks='area')
time.sleep(timesheet)

print("Starting task: volume")
experiment.Runexp1(num_images=num_images, model_instances=model_instances, tasks='volume')
time.sleep(timesheet)

# Run experiments for each task with sleep intervals between them
print("Starting task: position_common_scale")
experiment.Runexp1(num_images=num_images, model_instances=model_instances, tasks='position_common_scale')
time.sleep(timesheet)

print("Starting task: position_non_aligned_scale")
experiment.Runexp1(num_images=num_images, model_instances=model_instances, tasks='position_non_aligned_scale')
time.sleep(timesheet)

print("Starting task: length")
experiment.Runexp1(num_images=num_images, model_instances=model_instances, tasks='length')
time.sleep(timesheet)

print("Starting task: direction")
experiment.Runexp1(num_images=num_images, model_instances=model_instances, tasks='direction')
time.sleep(timesheet)

print("Starting task: angle")
experiment.Runexp1(num_images=num_images, model_instances=model_instances, tasks='angle')
time.sleep(timesheet)

print("Starting task: curvature")
experiment.Runexp1(num_images=num_images, model_instances=model_instances, tasks='curvature')
time.sleep(timesheet)

print("Starting task: shading")
experiment.Runexp1(num_images=num_images, model_instances=model_instances, tasks='shading')
