import os, sys
sys.path.append(os.path.abspath('customLLaVA/'))
import subprocess
import re
import numpy as np
import requests
import png
import io
import tempfile
import base64

class CustomLLaVA:

    @staticmethod
    def query(question, image):

        size = image.shape[0]
        grayscale = np.zeros((size,size), dtype=np.uint8)
        grayscale[image==0] = 255
        grayscale[image==1] = 0

        png_image = png.from_array(grayscale, 'L')

        MODEL_PATH = "../LLMP/customLLaVA/llava/checkpoints/llava-v1.5-7b-lora"       
        MODEL_BASE = "liuhaotian/llava-v1.5-7b"
        PYTHON_SCRIPT = "../LLMP/customLLaVA/llava/eval/run_llava.py"
        QUERY = question

        with tempfile.NamedTemporaryFile(suffix='.png', delete=True) as tmpfile:
            png_image.save(tmpfile.name)
            IMAGE_FILE = tmpfile.name

            command = f"python {PYTHON_SCRIPT} --model-path {MODEL_PATH} --model-base {MODEL_BASE} --image-file {IMAGE_FILE} --query '{question}'"

            output = subprocess.run(command, shell=True, capture_output=True, text=True)
    
            output_text = output.stdout
            #result = re.search(r'(\d+)(?:\.\d+)?$', output_text)
            result = re.search(r'(\d+(?:\.\d+)?)$', output_text)

            if result:
                extracted_number = result.group(1)
                return extracted_number
            else:
                print(output)
                """
                while not result:
                    print("Number not found.")
                    print(output)
                    output = subprocess.run(command, shell=True, capture_output=True, text=True) # run again
                    output_text = output.stdout
                    result = re.search(r'(\d+(?:\.\d+)?)$', output_text)
                extracted_number = result.group(1)
                return extracted_number"""
                    


            

