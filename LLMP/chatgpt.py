import sys
sys.path.append('./LLMP/')
import requests
import cv2
import numpy as np
import base64

class ChatGPT:
    @staticmethod
    def query(question, image):

        size = image.shape[0]
        grayscale = np.zeros((size,size), dtype=np.uint8)
        grayscale[image==0] = 255
        grayscale[image==1] = 0

        rgb = np.stack((grayscale,grayscale,grayscale),axis=-1)
        _, png = cv2.imencode('.png', rgb)

        base64_image = base64.b64encode(png.tobytes()).decode('utf-8')

        # OpenAI API Key
        api_key = "***********************************"

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
        "max_tokens": 1500
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        content_string = response.json()['choices'][0]['message']['content']
        return content_string  

