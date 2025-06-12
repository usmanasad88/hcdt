# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import os
from google import genai
from google.genai import types
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.imgutils import image_to_base64

def generate():
    image_data_b64_string = image_to_base64("/home/mani/Central/Cooking1/combined_frames/frame-00001.jpg")
        # Construct paths to the JSON files relative to this script's location
    script_dir = os.path.dirname(__file__)
    dag_file_path = os.path.join(script_dir, '..', 'data', 'Cooking', 'dag_noodles_v2.json')
    state_file_path = os.path.join(script_dir, '..', 'data', 'Cooking', 'state_noodles.json')
    with open(dag_file_path, 'r') as f:
            task_graph_string = f.read()
    with open(state_file_path, 'r') as f:
            state_schema_string = f.read()
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.5-flash-preview-05-20"
    # model = "gemini-2.0-flash"
    prompt_text = f"""
You are an AI assistant analyzing a cooking video frame.
Your goal is to update the state variables based on the provided task graph, state schema, and the visual information from the image.

Task Graph Definition:
```json
{task_graph_string}
```

State Variables Schema:
```json
{state_schema_string}
```

Instruction:
Based on the image provided and the schemas above, update the state variables.
Boolean state variables should be initialized as false if their state cannot be determined or is not true from the image.
Output the updated state variables as a JSON object.
Ensure your output is only the JSON object representing the updated state variables.
"""
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt_text),
                types.Part.from_bytes(
                    mime_type="image/jpeg",
                    data=base64.b64decode(image_data_b64_string),
                ),
            ],
        )
    ]

    generate_content_config = types.GenerateContentConfig(
        # response_mime_type="text/plain",
        response_mime_type="application/json",

    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")

if __name__ == "__main__":
    generate()
