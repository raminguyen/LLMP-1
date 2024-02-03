import os, sys
sys.path.append(os.path.abspath('LLaVA/'))

from io import StringIO
import numpy as np


class LLaVA:

    @staticmethod
    def query(question, image, model_path="liuhaotian/llava-v1.5-7b"):

        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        from llava.eval.run_llava import eval_model

        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path)
        )


        size = image.shape[0]
        grayscale = np.zeros((size,size), dtype=np.uint8)
        grayscale[image==0] = 255
        grayscale[image==1] = 0

        png_image = png.from_array(grayscale, 'L')

        outp = io.BytesIO()
        png_image.write(outp)
        png_bytes = outp.getvalue()
        outp.close()

        base64_image = base64.b64encode(png_bytes).decode('utf-8')


        # prompt = "What are the things I should be cautious about when I visit here?"
        # image_file = "https://llava-vl.github.io/static/images/view.jpg"

        args = type('Args', (), {
            "model_path": model_path,
            "model_base": None,
            "model_name": get_model_name_from_path(model_path),
            "query": question,
            "conv_mode": None,
            "image_file": f"data:image/png;base64,{base64_image}",
            "sep": ",",
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 512
        })()

        # pipe std out
        result = StringIO()
        old_stdout = sys.stdout
        sys.stdout = result

        eval_model(args)

        sys.stdout = old_stdout

        return result.getvalue()

