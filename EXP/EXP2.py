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
    #"CustomLLaMA": L.llamafinetuned("./outputEXP2-5000/json/finetuned_llama_EXP2"),
    "LLaMA": L.llama("meta-llama/Llama-3.2-11B-Vision-Instruct"),
    "GeminiProVision": L.GeminiProVision(),
    "Gemini1_5Flash": L.Gemini1_5Flash()
}

# Set up the number of images and sleep time
runexp2_num_images = 2
runexp2_timesheet = 1  # Sleep for 5 seconds between each task

# Create an instance of the Runexp2 class
runexp2_experiment = L.Runexp2()

# Run experiment for "bar" task
print("Starting task: bar")
runexp2_experiment.Runexp2(num_images=runexp2_num_images, model_instances=model_instances, tasks="bar")
time.sleep(runexp2_timesheet)

#Run experiment for "pie" task
print("Starting task: pie")
runexp2_experiment.Runexp2(num_images=runexp2_num_images, model_instances=model_instances, tasks="pie")
time.sleep(runexp2_timesheet)

print("All tasks completed. Yeah!")