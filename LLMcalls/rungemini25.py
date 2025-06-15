# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import os
from google import genai
from google.genai import types
import sys
import os
from typing import Optional, List, Dict
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.imgutils import image_to_base64
from utils.textutils import extract_json_from_response

def generate(image_paths: List[str], 
    query_prompt: str, 
    responses: Optional[List[str]] = None):
    encoded_images = []
    for img_path in image_paths:
        encoded_img = image_to_base64(img_path, target_largest_dimension=None)
        encoded_images.append(encoded_img) # Will store None if encoding failed
    
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    # model = "gemini-2.5-flash-preview-05-20"
    model = "gemma-3-27b-it"
    # model = "gemini-2.0-flash"

    current_responses = responses if responses is not None else []
    contents = []

    # Build context from provided image-response pairs (history)
    for i in range(len(image_paths)):
        prompt_for_turn = query_prompt if i == 0 else "Next frame:"
        
        user_parts = [types.Part.from_text(text=prompt_for_turn)]
        if i < len(encoded_images) and encoded_images[i]:
            try:
                image_bytes = base64.b64decode(encoded_images[i])
                user_parts.append(types.Part.from_bytes(
                    mime_type="image/jpeg", # Assuming JPEG, adjust if type varies
                    data=image_bytes,
                ))
            except Exception as e:
                print(f"Warning: Could not decode or use image {image_paths[i]} for history turn {i}: {e}")
        
        contents.append(types.Content(role="user", parts=user_parts))
        # contents.append(types.Content(role="model", parts=[types.Part.from_text(current_responses[i])]))


    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
        temperature=0,  
        top_p=1,        
        top_k=50 
        # response_mime_type="application/json",

    )
    full_response = []
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if chunk.text is not None: # Check if chunk.text is not None
            # print(chunk.text, end="")
            full_response.append(chunk.text)

    return "".join(full_response)

if __name__ == "__main__":
    case_study="HAViD"
    script_dir = os.path.dirname(__file__)
    dag_file_path = os.path.join(script_dir, '..', 'data', case_study, 'dag.json')
    state_file_path = os.path.join(script_dir, '..', 'data', case_study, 'state.json')
    result_store_path= os.path.join(script_dir, '..', 'data', case_study, 'result_test.json')
    with open(dag_file_path, 'r') as f:
            task_graph_string = f.read()
    with open(state_file_path, 'r') as f:
            state_schema_string = f.read()

    prompt_text = f"""
    You are an AI assistant analyzing video frames of a person performing a task.
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
    Boolean state variables are not strictly boolean and can be False, True or Unknown.
    Output the updated state variables as a JSON object.
    Ensure your output is only the JSON object representing the updated state variables.
    """
    base_image_dir = "/home/mani/Central/HaVid/S01A02I01S1/"
    total_frames_to_process = 'all'  # Set the total number of frames you want to process (e.g., frames 1 to 100)
    if total_frames_to_process == 'all':
        total_frames_to_process = len([f for f in os.listdir(base_image_dir) if f.endswith('.png')])
    all_responses_data = []
    current_response = None

    for frame_num in range(1, total_frames_to_process + 1):
        current_frame_path = os.path.join(base_image_dir, f"frame_{frame_num:04d}.png")

        if frame_num == 1:
            current_response = generate([current_frame_path], prompt_text)
            response_obj = json.loads(extract_json_from_response(current_response))     
            all_responses_data.append({"frame": frame_num, "state": response_obj})        

        else:
            prev_frame = frame_num - 1
            prev_frame_path = os.path.join(base_image_dir, f"frame_{prev_frame:04d}.png")

            call_image_paths = [prev_frame_path, current_frame_path]
            call_responses = [current_response]
            current_response = generate(call_image_paths, prompt_text, responses=call_responses)              
            response_obj = json.loads(extract_json_from_response(current_response))
            all_responses_data.append({"frame": frame_num, "state": response_obj})        
        
        all_responses_data.sort(key=lambda x: x.get("frame", float('inf'))) # Sort by frame number
        with open(result_store_path, 'w') as f:
            json.dump(all_responses_data, f, indent=4)