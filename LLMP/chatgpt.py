import requests
import png
import io
import numpy as np
import base64

class ChatGPT:
    @staticmethod
    def query(question, image):

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

        # OpenAI API Key
        api_key = "***************"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": question
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
                }
            ]
            }
        ],
        "max_tokens": 1600
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        response_json = response.json()

        if ('error' in response_json):
            print('ERROR', response_json)

        if 'choices' in response_json:
            content_string = response_json['choices'][0]['message']['content']
            return content_string  

