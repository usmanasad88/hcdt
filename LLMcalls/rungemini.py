import subprocess
from PIL import Image
import matplotlib.pyplot as plt
import json
import re


import os
import subprocess

# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import os
from google import genai
from google.genai import types


def generate():
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.5-flash-preview-04-17"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""INSERT_INPUT_HERE"""),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")


for timestamp in range(0, 42):
    with open("/home/mani/CLoSD/closd/IntentNet/prompt.txt", "r") as f:
        prompt = f.read()
    image_path = f"/home/mani/Central/Cooking1/Stack/output_frames/second/cam2_cr-frame-{timestamp}.0.jpg"
    response = generate(
        prompt=prompt,
        image_path=image_path
    )
    if response:        
        with open("/home/mani/CLoSD/closd/IntentNet/prompt.txt", "a") as f:
            f.write(f"<start_of_turn>model\n{response}<end_of_turn>\n")
            f.write(f"<start_of_turn>user\nThe next image at time {timestamp+1}<end_of_turn>\n")
       
