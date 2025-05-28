import os
import mimetypes
import json # Added import
from google import genai
from google.genai import types
import base64


def save_binary_file(file_name, data):
    with open(file_name, "wb") as f:
        f.write(data)
    print(f"File saved to to: {file_name}")

# generate_history function is removed as state is managed directly via JSON.

def gemini_generate(prompt, image_path):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    uploaded_image = client.files.upload(file=image_path)
    # Using a model that is good with JSON and text generation.
    # Consider "gemini-1.5-flash-latest" or "gemini-1.5-pro-latest" if available and suitable.
    # "gemini-2.0-flash-preview-image-generation" might be an older or specialized name.
    # For now, keeping the user's model, but this is a key point for JSON reliability.
    model = "gemini-2.0-flash-preview-image-generation" # Changed to a generally available model good with JSON

    system_prompt = """Point to the gearbox with no more than 10 items. The answer should follow the json format: [{"point": <point>, "label": <label1>}, ...]. The points are in [y, x] format normalized to 0-1000."""

    contents = [
        types.Content( # System prompt defining role, task, and output format
            role="user",
            parts=[types.Part.from_text(text=system_prompt)]
        ),
        
        ]

 
    # Add the current user turn (image and the prompt containing the current state)
    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_uri(
                    file_uri=uploaded_image.uri,
                    mime_type=uploaded_image.mime_type,
                ),
                types.Part.from_text(text=prompt), # This contains the current state JSON
            ],
        )
    )

    generate_content_config = types.GenerateContentConfig(
        response_modalities=["IMAGE", "TEXT"], 
        response_mime_type="text/plain", 
    )

    full_text = ""
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
            full_text += chunk.text if hasattr(chunk, "text") else str(chunk.candidates[0].content.parts[0].text)
    return full_text

   
def gemma_api_generate(prompt, image_path):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()

    # data=base64.b64encode(image_data)

    model = "gemma-3-12b-it"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(
                    mime_type="image/png",
                    data=image_data,
                    ),
                
                types.Part.from_text(text="""Point to the gearbox. The answer should follow the json format: [{\"point\": <point>, \"label\": <label1>}, ...]. The points are in [y, x] format normalized to 0-1000."""),
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

if __name__ == "__main__":
    prompt = "Please point to the gearbox in the image."
    image_path = "/home/mani/Central/HaVid/S01A02I01S1/frame_0001.png"  
    # Gearbox location:
    # Image Size: 1280x720, Gearbox at (875, 574) pixels
    # Normalized to 0-1000 (y: 574/720*1000, x: 875/1280*1000)
    # Normalized coordinates: [798.6111111111111, 683.59375]    
    llm_response_str = gemma_api_generate(prompt, image_path)

            
            