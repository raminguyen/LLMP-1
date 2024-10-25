import sys
import os
import torch
from dotenv import load_dotenv
sys.path.append('../')
import LLMP as L

load_dotenv()
torch.cuda.empty_cache()

# Import LLMP after ensuring the path is correct
import LLMP as L

# Hugging Face login using the token
from huggingface_hub import login
login('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')

model_instances = {
   #"gpt4o": L.GPTModel("gpt-4o"),
    "CustomLLaMA": L.llamafinetuned("/home/huuthanhvy.nguyen001/LLMP/EXP/my_finetuned_llama_7200_images"),
    #"LLaMA": L.llama("meta-llama/Llama-3.2-11B-Vision-Instruct"),
    #"GeminiProVision": L.GeminiProVision(),
    #"Gemini1_5Flash": L.Gemini1_5Flash()
}

# Run the evaluator
e = L.Evaluator()

import time

# Define the query
bestquery = """
What is the exact acute angle degree? Give your answer as a specific number.
No extra explanation is required.

"""

# Generate images
images = [L.GPImage.figure1('angle') for i in range(10)]

# Start time measurement
start_time = time.time()

# Run the evaluator
result = e.run(images, bestquery, model_instances)

# End time measurement
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")

# Save results to JSON file
e.save_results('RESULTS/test.json')

# Save elapsed time to a text file
with open('RESULTS/test', 'w') as f:
    f.write(f"Elapsed time for running the query: {elapsed_time} seconds\n")