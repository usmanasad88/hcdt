import google.generativeai as genai
from PIL import Image
import os
from typing import Optional, Union, List
import json

# Ensure GOOGLE_API_KEY is set or genai.configure() is called before use.
# Example:
# if os.getenv("GOOGLE_API_KEY"):
#     genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# else:
#     print("Warning: GOOGLE_API_KEY not set. Token counting may fail or use default credentials.")


def count_gemini_tokens_simple(
    model_name: str,
    text_input: Optional[str] = None,
    image_input: Optional[Union[str, Image.Image]] = None
) -> int:
    """
    Counts the number of tokens for a given text and/or image input
    for a specified Gemini model (simplified version).

    Args:
        model_name (str): The name of the Gemini model (e.g., "gemini-1.5-flash-latest").
        text_input (Optional[str]): The text input.
        image_input (Optional[Union[str, Image.Image]]):
            The image input (file path or PIL.Image.Image object).

    Returns:
        int: The total number of tokens.
    """
    
    model = genai.GenerativeModel(model_name)
    parts: List[Union[str, Image.Image]] = []

    if text_input:
        parts.append(text_input)

    if image_input:
        if isinstance(image_input, str):
            img = Image.open(image_input)
            parts.append(img)
        elif isinstance(image_input, Image.Image):
            parts.append(image_input)
        # No explicit error for invalid image type in this simplified version

    if not parts:
        return 0 # No content, no tokens

    response = model.count_tokens(parts)
    return response.total_tokens

# --- DAG Helper Functions ---
# Ensure dag_noodles.json is in /home/mani/Repos/hcdt/data/
DAG_FILE_PATH = "/home/mani/Repos/hcdt/data/dag_noodles.json"

def load_dag(filepath=DAG_FILE_PATH):
    """Loads the DAG from a JSON file and initializes step statuses."""
    try:
        with open(filepath, 'r') as f:
            dag_data = json.load(f)
        # Initialize status for each step and ensure 'predecessors' key exists
        for step in dag_data:
            step['status'] = 'pending'  # Possible statuses: 'pending', 'available', 'completed'
            if 'predecessors' not in step or not step['predecessors']:
                step['predecessors'] = []
        return dag_data
    except FileNotFoundError:
        print(f"Error: DAG file not found at {filepath}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        return []

def get_available_steps(dag_steps):
    """
    Determines which steps are available based on completed predecessors.
    Marks them as 'available' in the main dag_steps list and returns a list of available step info.
    """
    available_for_prompt = []
    completed_ids = {step['id'] for step in dag_steps if step['status'] == 'completed'}

    for step in dag_steps:
        if step['status'] == 'pending':
            # Check if all predecessors are completed
            if not step['predecessors'] or all(pid in completed_ids for pid in step['predecessors']):
                step['status'] = 'available' # Update status in the main list
                available_for_prompt.append({"id": step["id"], "description": step["description"]})
        elif step['status'] == 'available': # If already marked available, ensure it's still in the list for LLM
            if not step['predecessors'] or all(pid in completed_ids for pid in step['predecessors']):
                 available_for_prompt.append({"id": step["id"], "description": step["description"]})
            else: # Should not happen if logic is correct, but as a safeguard
                step['status'] = 'pending'


    return available_for_prompt

def update_step_status(dag_steps, step_id, new_status):
    """Updates the status of a specific DAG step in the list."""
    for step in dag_steps:
        if step['id'] == step_id:
            if step['status'] == 'completed' and new_status != 'completed':
                print(f"Warning: Attempting to change status of already completed step '{step_id}' to '{new_status}'. Allowing for now.")
            step['status'] = new_status
            return True
    print(f"Warning: Step ID '{step_id}' not found in DAG for status update.")
    return False

def get_step_description_by_id(dag_steps, step_id):
    """Gets the description of a step by its ID."""
    for step in dag_steps:
        if step['id'] == step_id:
            return step.get("description", "Unknown step")
    return "Unknown step"

# if __name__ == '__main__':
#     env_api_key = os.getenv("GOOGLE_API_KEY")
#     if env_api_key:
#         genai.configure(api_key=env_api_key)
#         print(f"Configured Gemini API with key ending in: ...{env_api_key[-4:]}")
#     else:
#         print("Warning: GOOGLE_API_KEY not set. The __main__ example for token counting might not work.")
#         print("Please set the GOOGLE_API_KEY environment variable.")

#     if env_api_key: # Only run example if API key is likely configured
#         test_model = "gemini-1.5-flash-latest" # Use a common, efficient model for testing

#         # 1. Text only
#         text_only_tokens = count_gemini_tokens_simple(
#             model_name=test_model,
#             text_input="This is a sample text for token counting."
#         )
#         print(f"Tokens for text only: {text_only_tokens}")

#         # Create a dummy image for testing
#         dummy_image_path = "dummy_for_token_count.png"
#         try:
#             img = Image.new('RGB', (60, 30), color = 'red')
#             img.save(dummy_image_path)

#             # 2. Image path only
#             image_path_tokens = count_gemini_tokens_simple(
#                 model_name=test_model,
#                 image_input=dummy_image_path
#             )
#             print(f"Tokens for image (from path) only: {image_path_tokens}")

#             # 3. PIL Image object only
#             pil_image = Image.open(dummy_image_path)
#             image_pil_tokens = count_gemini_tokens_simple(
#                 model_name=test_model,
#                 image_input=pil_image
#             )
#             print(f"Tokens for image (PIL object) only: {image_pil_tokens}")
#             pil_image.close()

#             # 4. Both text and image path
#             text_and_image_tokens = count_gemini_tokens_simple(
#                 model_name=test_model,
#                 text_input="Describe this image.",
#                 image_input=dummy_image_path
#             )
#             print(f"Tokens for text and image (from path): {text_and_image_tokens}")

#         except Exception as e:
#             print(f"An error occurred during the __main__ example: {e}")
#             print("This might be due to missing libraries (Pillow) or API key issues.")
#         finally:
#             if os.path.exists(dummy_image_path):
#                 os.remove(dummy_image_path)
#     else:
#         print("Skipping __main__ example execution as GOOGLE_API_KEY is not detected.")