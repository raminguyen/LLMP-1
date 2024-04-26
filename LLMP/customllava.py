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
import runpy
import sys
from io import StringIO
from contextlib import redirect_stdout

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

            #command = f"python {PYTHON_SCRIPT} --model-path {MODEL_PATH} --model-base {MODEL_BASE} --image-file {IMAGE_FILE} --query '{question}'"

            # Define the script path and the arguments as globals
            globals_dict = {
                "argv": [
                    PYTHON_SCRIPT,
                    "--model-path", MODEL_PATH,
                    "--model-base", MODEL_BASE,
                    "--image-file", IMAGE_FILE,
                    "--query", QUERY
                ]
            }

            """
            # Capture the printed output
            output_capture = StringIO()
            sys.stdout = output_capture
            
            # Temporarily replace the system argv
            original_argv = sys.argv
            sys.argv = globals_dict['argv']
            
            # Run the script
            result = runpy.run_path(PYTHON_SCRIPT, run_name="__main__")
            
            # Restore the original argv and stdout
            sys.argv = original_argv
            sys.stdout = sys.__stdout__
            
            # Get the captured output
            captured_output = output_capture.getvalue()
            output_text = captured_output.strip().split('\n')[-1]
            
            """

             # Capture the printed output
            with io.StringIO() as output_capture:
                with redirect_stdout(output_capture):
                    # Temporarily replace the system argv
                    original_argv = sys.argv
                    sys.argv = globals_dict['argv']
                    
                    # Run the script
                    result = runpy.run_path(PYTHON_SCRIPT, run_name="__main__")
                    
                    # Restore the original argv
                    sys.argv = original_argv
                    
                # Get the captured output
                captured_output = output_capture.getvalue().strip().split('\n')[-1]


            return captured_output

            """
            #result = re.search(r'(\d+(?:\.\d+)?)$', output_text)
            result = re.search(r'(\d+(?:\.\d+)?)$', captured_output)
            if result:
                extracted_number = result.group(1)
                return extracted_number
            else:
                print(captured_output)"""

                    


            

