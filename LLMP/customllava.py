import os
import sys
import numpy as np
import png
import tempfile
import runpy
import io
from contextlib import redirect_stdout, redirect_stderr

class CustomLLaVA:
    @staticmethod
    def query(question, image):
        size = image.shape[0]
        grayscale = np.zeros((size,size), dtype=np.uint8)
        grayscale[image==0] = 255
        grayscale[image==1] = 0
        png_image = png.from_array(grayscale, 'L')
        
        MODEL_PATH = "../LLMP/LLaVA/llava/checkpoints/llava-v1.5-7b-lora"       
        MODEL_BASE = "liuhaotian/llava-v1.5-7b"
        PYTHON_SCRIPT = "../LLMP/LLaVA/llava/eval/run_llava.py"
        QUERY = question

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
            png_image.save(tmpfile.name)
            IMAGE_FILE = tmpfile.name

            globals_dict = {
                "argv": [
                    PYTHON_SCRIPT,
                    "--model-path", MODEL_PATH,
                    "--model-base", MODEL_BASE,
                    "--image-file", IMAGE_FILE,
                    "--query", QUERY
                ]
            }

            try:
                # Capture both stdout and stderr
                with io.StringIO() as output_capture, io.StringIO() as error_capture:
                    with redirect_stdout(output_capture), redirect_stderr(error_capture):
                        # Temporarily replace the system argv
                        original_argv = sys.argv
                        sys.argv = globals_dict['argv']
                        
                        # Run the script
                        result = runpy.run_path(PYTHON_SCRIPT, run_name="__main__")
                        
                        # Restore the original argv
                        sys.argv = original_argv
                    
                    # Get the captured output and errors
                    captured_output = output_capture.getvalue().strip()
                    #captured_errors = error_capture.getvalue().strip()

                # Check if there were any errors
                #if captured_errors:
                    #print(f"Errors occurred:\n{captured_errors}")
                    #return None

                # Process the output
                output_lines = captured_output.split('\n')
                if output_lines:
                    return output_lines[-1]  # Return the last line of output
                else:
                    print("No output was captured.")
                    return None

            except Exception as e:
                print(f"An error occurred: {str(e)}")
                return None

            finally:
                # Clean up the temporary file
                os.unlink(IMAGE_FILE)
