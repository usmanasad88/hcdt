import subprocess
from PIL import Image
import matplotlib.pyplot as plt
import json
import re
import requests
from typing import Optional, List, Dict
import sys
import os
import base64

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.imgutils import encode_file_to_base64, image_to_base64

OLLAMA_API_BASE_URL = "http://localhost:11434/api"

def encode_file_to_base64_local(file_path):
    """Encodes a file to a base64 string."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None
    try:
        with open(file_path, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode("utf-8")
        return encoded_string
    except Exception as e:
        print(f"Error encoding file {file_path}: {e}")
        return None

def run_ollama_messages(messages: List[Dict], model: str = "llama3.2-vision", stream: bool = False) -> str:
    """
    Sends messages with images to Ollama using the chat API for in-context learning.
    
    Args:
        messages (List[Dict]): List of message dictionaries with role, content, and optional images
        model (str): The model name to use
        stream (bool): Whether to stream the response
        
    Returns:
        str: The complete response content
    """
    chat_endpoint = f"{OLLAMA_API_BASE_URL}/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream
    }
    
    try:
        response = requests.post(chat_endpoint, json=payload, stream=stream)
        response.raise_for_status()
        
        if stream:
            full_response_content = ""
            for line in response.iter_lines():
                if line:
                    try:
                        json_chunk = json.loads(line)
                        if "message" in json_chunk and "content" in json_chunk["message"]:
                            print(json_chunk["message"]["content"], end='', flush=True)
                            full_response_content += json_chunk["message"]["content"]
                        elif json_chunk.get("done"):
                            print("\n\n--- Generation complete ---")
                            break
                    except json.JSONDecodeError:
                        print(f"Could not decode JSON from line: {line.decode('utf-8')}")
            return full_response_content
        else:
            response_json = response.json()
            return response_json["message"]["content"]
            
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Ollama: {e}")
        print(f"Ensure '{model}' model is pulled and Ollama is running.")
        return ""

def create_cooking_context_messages(image_paths: List[str], query_prompt: str) -> List[Dict]:
    """
    Creates a context for cooking video analysis with example messages.
    
    Args:
        image_paths (List[str]): List of paths to cooking video frames
        query_prompt (str): The actual question to ask about the cooking process
        
    Returns:
        List[Dict]: Formatted messages for in-context learning
    """
    # Sample a few frames for context (use first, middle, and last frames)
    context_frames = []
    if len(image_paths) >= 3:
        context_frames = [
            image_paths[0],  # First frame
            image_paths[len(image_paths)//2],  # Middle frame
            image_paths[-1]  # Last frame
        ]
    else:
        context_frames = image_paths
    
    # Encode context images
    encoded_images = []
    for img_path in context_frames:
        encoded_img = image_to_base64(img_path, target_largest_dimension=512)  # Smaller for context
        if encoded_img:
            encoded_images.append(encoded_img)
    
    # Encode all images for the main query (sample every 10th frame to avoid overwhelming)
    query_images = []
    step = max(1, len(image_paths) // 20)  # Sample up to 20 frames
    for i in range(0, len(image_paths), step):
        encoded_img = image_to_base64(image_paths[i], target_largest_dimension=512)
        if encoded_img:
            query_images.append(encoded_img)
    
    messages = [
        # Example 1: Identifying cooking action
        {
            "role": "user",
            "content": "What cooking action is happening in this frame?",
            "images": [encoded_images[0]] if encoded_images else []
        },
        {
            "role": "assistant",
            "content": "In this frame, I can see the chef preparing ingredients. They appear to be chopping vegetables on a cutting board, which is a preparatory step in the cooking process."
        },
        # Example 2: Describing cooking progression
        {
            "role": "user", 
            "content": "How has the cooking progressed in this frame compared to earlier?",
            "images": [encoded_images[1]] if len(encoded_images) > 1 else []
        },
        {
            "role": "assistant",
            "content": "The cooking has progressed significantly. The chef is now actively cooking at the stove, with ingredients being heated in a pan. The preparation phase has moved to the active cooking phase."
        },
        # Example 3: Final dish identification
        {
            "role": "user",
            "content": "What is the final result shown in this cooking frame?",
            "images": [encoded_images[2]] if len(encoded_images) > 2 else []
        },
        {
            "role": "assistant", 
            "content": "This frame shows the completed dish being plated. The chef has finished cooking and is now presenting the final prepared meal."
        },
        # The actual query
        {
            "role": "user",
            "content": query_prompt,
            "images": query_images
        }
    ]
    
    return messages

def create_context_messages(
    image_paths: List[str], 
    query_prompt: str, 
    responses: Optional[List[str]] = None
) -> List[Dict]:
    """
    Creates a message history for Ollama.
    It includes past interactions if `responses` are provided, and a final user query 
    for the last image in `image_paths`.

    The `query_prompt` is used for the first user turn in the history, or for the
    sole user turn if no history (`responses`) is provided. Subsequent user turns
    in the history and the final query (if history exists) use "Next frame:".

    Args:
        image_paths (List[str]): List of paths to images. 
                                 Its length must be `len(responses) + 1`.
        query_prompt (str): The text prompt for the first turn in the sequence.
        responses (Optional[List[str]]): Optional list of assistant responses for the 
                                         initial `len(responses)` images.
        
    Returns:
        List[Dict]: Formatted messages for in-context learning.
        
    Raises:
        ValueError: If `len(image_paths)` is not equal to `len(responses) + 1`.
    """
    
    if responses is None:
        current_responses = []
    else:
        current_responses = responses

    # if len(image_paths) != len(current_responses) + 1:
    #     raise ValueError(
    #         f"Input length mismatch: len(image_paths) ({len(image_paths)}) "
    #         f"must be len(responses) ({len(current_responses)}) + 1."
    #     )

    messages = []
    
    encoded_images = []
    for img_path in image_paths:
        encoded_img = image_to_base64(img_path, target_largest_dimension=512)
        encoded_images.append(encoded_img) # Will store None if encoding failed

    # Build context from provided image-response pairs (history)
    for i in range(len(image_paths)):
        user_prompt_text = query_prompt if i == 0 else "Next frame:"

        user_message = {"role": "user", "content": user_prompt_text}
        if i < len(encoded_images) and encoded_images[i]: # Check index and if encoding was successful
            user_message["images"] = [encoded_images[i]]
        messages.append(user_message)
        
        # messages.append({"role": "assistant", "content": current_responses[i]})

    
    return messages



# Example usage with your existing image paths
if __name__ == "__main__":
    # Use your existing image path setup
    script_dir = os.path.dirname(__file__)
    dag_file_path = os.path.join(script_dir, '..', 'data', 'Cooking', 'dag_noodles_v2.json')
    state_file_path = os.path.join(script_dir, '..', 'data', 'Cooking', 'state_noodles.json')
    with open(dag_file_path, 'r') as f:
            task_graph_string = f.read()
    with open(state_file_path, 'r') as f:
            state_schema_string = f.read()

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
Boolean state variables are not strictly boolean, and can be True, False or Unknown.
Output the updated state variables as a JSON object.
"""
    image_paths = []
    base_path = "/home/mani/Central/Cooking1/combined_frames/"
    num_frames = 570
    
    for i in range(1, num_frames + 1):
        frame_number_str = str(i).zfill(5)
        image_path = f"{base_path}frame-{frame_number_str}.jpg"
        if os.path.exists(image_path):  # Only add existing files
            image_paths.append(image_path)

    for i in range(1, 11):
        sample_frames = image_paths[:i] if len(image_paths) >= 10 else image_paths
        if sample_frames:
            simple_messages = create_context_messages(sample_frames, prompt_text)
            
            print("\n\n--- Simple Analysis ---")
            simple_response = run_ollama_messages(simple_messages, model="gemma3:12b", stream=False)
            print(f"Simple analysis result: {simple_response}")

# Utility functions from your original file
