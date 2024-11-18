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

model = {
    "gpt4o": L.GPTModel("gpt-4o"), 
    #"CustomLLaMA": L.llamafinetuned("./finetuning-EXP1-100-10epochs-0.001-test/fine_tuned_model"),
    #"LLaMA": L.llama("meta-llama/Llama-3.2-11B-Vision-Instruct"),
    "GeminiProVision": L.GeminiProVision(),
    "Gemini1_5Flash": L.Gemini1_5Flash()
}

# Example of checking for model-specific method
try:
    trainable_params = model.get_trainable_params()  # Hypothetical method
    print(f"{model_name}: {trainable_params:,} trainable parameters")
except AttributeError:
    print(f"{model_name}: Model does not support parameter inspection.")
