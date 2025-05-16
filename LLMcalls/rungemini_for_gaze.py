import subprocess
from PIL import Image
import matplotlib.pyplot as plt
import json
import re


import os
import subprocess

# To run this code you need to install the following dependencies:
# pip install google-genai

# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import mimetypes
import os
from google import genai
from google.genai import types


def save_binary_file(file_name, data):
    f = open(file_name, "wb")
    f.write(data)
    f.close()
    print(f"File saved to to: {file_name}")


def generate(prompt, image_path):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    files = [
        # Please ensure that the file is available in local system working direrctory or change the file path.
        client.files.upload(file="data/combined_gaze/combined-frame-0.0.jpg"),
        client.files.upload(file=image_path),
    ]
    model = "gemini-2.0-flash-preview-image-generation"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_uri(
                    file_uri=files[0].uri,
                    mime_type=files[0].mime_type,
                ),
                types.Part.from_text(text="""Using the provided image and gaze heatmap, determine the most likely object of the person's gaze. Output your prediction as a single word. Possible outputs include object names, \"camera\", or \"unclear\".
"""),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="""Table"""),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_uri(
                    file_uri=files[1].uri,
                    mime_type=files[1].mime_type,
                ),
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_modalities=[
            "IMAGE",
            "TEXT",
        ],
        response_mime_type="text/plain",
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if (
            chunk.candidates is None
            or chunk.candidates[0].content is None
            or chunk.candidates[0].content.parts is None
        ):
            continue
        if chunk.candidates[0].content.parts[0].inline_data:
            file_name = "ENTER_FILE_NAME"
            inline_data = chunk.candidates[0].content.parts[0].inline_data
            data_buffer = inline_data.data
            file_extension = mimetypes.guess_extension(inline_data.mime_type)
            save_binary_file(f"{file_name}{file_extension}", data_buffer)
        else:
            print(chunk.text)
            return chunk.text

if __name__ == "__main__":
    # prompt="""Using the provided image and gaze heatmap, determine the most likely object of the person's gaze. Output your prediction as a single word. Possible outputs include object names, "camera", or "unclear"."""
    # results_store_file="LLMcalls/gazetargets.txt"
         
    # with open(results_store_file, "w") as f_out:
    #     for second in range(0, 42):
    #         for fractionsecond in range(0, 5):
    #             timestamp = f"{second:01d}.{2*fractionsecond:01d}"
    #             image_path = f"/home/mani/Central/Cooking1/Stack/cam1/output_combined_frames/combined-frame-{timestamp}.jpg"
    #             response = generate(prompt, image_path)
                
                
    #             print(f"{timestamp}\t{response}\n")
    #             # Store the raw responses
    #         f_out.write(f"{timestamp}\t{response}\n")