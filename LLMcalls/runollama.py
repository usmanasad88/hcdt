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

def analyze_cooking_video_sequence(image_paths: List[str], model: str = "gemma3:12b") -> str:
    """
    Analyzes a sequence of cooking video frames to determine what dish is being made.
    
    Args:
        image_paths (List[str]): List of paths to video frames
        model (str): Model to use for analysis
        
    Returns:
        str: Analysis of what the chef is making
    """
    query_prompt = """Analyze this sequence of cooking video frames and determine:
1. What dish is the chef making?
2. What are the main ingredients being used?
3. What cooking techniques are being employed?
4. What is the approximate cooking progression from start to finish?

Please provide a detailed analysis based on all the frames provided."""
    
    messages = create_cooking_context_messages(image_paths, query_prompt)
    
    print(f"--- Analyzing cooking sequence with {len(image_paths)} frames using {model} ---")
    return run_ollama_messages(messages, model=model, stream=True)

# Example usage with your existing image paths
if __name__ == "__main__":
    # Use your existing image path setup
    image_paths = []
    base_path = "/home/mani/Central/Cooking1/combined_frames/"
    num_frames = 570
    
    for i in range(1, num_frames + 1):
        frame_number_str = str(i).zfill(5)
        image_path = f"{base_path}frame-{frame_number_str}.jpg"
        if os.path.exists(image_path):  # Only add existing files
            image_paths.append(image_path)
    
    if image_paths:
        print(f"Found {len(image_paths)} frames to analyze")
        response = analyze_cooking_video_sequence(image_paths, model="gemma3:12b")
        print(f"\n\nFinal Analysis:\n{response}")
    else:
        print("No valid image frames found. Please check the base_path.")
    
    # Alternative: Simple question with fewer frames
    sample_frames = image_paths[:10] if len(image_paths) >= 10 else image_paths
    if sample_frames:
        simple_query = "What is the chef making in these cooking frames?"
        simple_messages = create_cooking_context_messages(sample_frames, simple_query)
        
        print("\n\n--- Simple Analysis ---")
        simple_response = run_ollama_messages(simple_messages, model="gemma3:12b", stream=False)
        print(f"Simple analysis result: {simple_response}")

# Utility functions from your original file
