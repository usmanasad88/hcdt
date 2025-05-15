import subprocess
from PIL import Image
import matplotlib.pyplot as plt
import json
import re
import requests

def run_ollama_llama_vision(prompt: str, image_path: str, model: str = "llama3.2-vision") -> dict:
    """
    Sends a prompt and image to the Ollama server running llama3.2-vision.
    Returns the model's response as a dict (parsed JSON).
    """
    url = "http://localhost:11434/api/generate"
    with open(image_path, "rb") as img_file:
        image_bytes = img_file.read()
    import base64
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "format": "json",
        "stream": False
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()["response"]  # This will be a dict if the model responds with JSON



# def run_llama_mtmd(prompt: str, image_path: str) -> str:
#     cmd = [
#         "/home/mani/Repos/llama.cpp/build/bin/llama-mtmd-cli",
#         "-m", "/home/mani/Repos/llama.cpp/models/gemma-3-12b-it-q4_0.gguf",
#         "--mmproj", "/home/mani/Repos/llama.cpp/models/mmproj-model-f16-12B.gguf",
#         "-p", prompt,
#         "--image", image_path
#     ]
    
#     try:
#         result = subprocess.run(cmd, capture_output=True, text=True, check=True)
#         output = result.stdout.strip()
#         return output
#     except subprocess.CalledProcessError as e:
#         print("Error during inference:")
#         print(e.stderr)
        # return None

def resize_image_to_model_size(image_path, w_model, h_model, save_path=None):
    """
    Resize the input image to the model's expected size.

    Args:
        image_path (str): Path to the input image.
        w_model (int): Target width for the model.
        h_model (int): Target height for the model.
        save_path (str, optional): If provided, saves the resized image to this path.
                                   Otherwise, returns the resized PIL Image object.

    Returns:
        PIL.Image.Image or str: The resized image object or the save path if saved.
    """
    img = Image.open(image_path)
    img_resized = img.resize((w_model, h_model), Image.LANCZOS)
    if save_path:
        img_resized.save(save_path)
        return save_path
    return img_resized

resize_image_to_model_size("/home/mani/Central/Cooking1/Stack/output_frames/cam1-frame-1.4.jpg", 896, 896, "/home/mani/Downloads/me_resized.jpg")


run_llama=True
response=None
if run_llama:    
    response = run_ollama_llama_vision(
        prompt="what is the object at x-pixel 440. y-pixel 438 in this image, given that the image is 1280x720 pixels.",
        image_path="/home/mani/Downloads/me_resized.jpg"
)

def plot_points_on_image(image_path, points, point_color='red', point_size=50):
    """
    Plots given (x, y) points on the image.

    Args:
        image_path (str): Path to the image file.
        points (list of tuple): List of (x, y) coordinates.
        point_color (str): Color of the points.
        point_size (int): Size of the points.
    """
    img = Image.open(image_path)
    plt.imshow(img)
    xs, ys = zip(*points)
    plt.scatter(xs, ys, c=point_color, s=point_size)
    plt.axis('off')
    plt.show()

# Example usage:
w_model = 896
h_model = 896
with Image.open("/home/mani/Downloads/me_resized.jpg") as img:
    w_orig, h_orig = img.size

def extract_xy_coords(data):
    """
    Recursively extract all dicts with 'x' and 'y' keys from data.
    Returns a list of dicts with 'x' and 'y'.
    """
    coords = []
    if isinstance(data, dict):
        # If this dict has x and y, add it
        if "x" in data and "y" in data:
            coords.append(data)
        # Otherwise, search its values
        for v in data.values():
            coords.extend(extract_xy_coords(v))
    elif isinstance(data, list):
        for item in data:
            coords.extend(extract_xy_coords(item))
    return coords

if response:
    print("Model output:")
    print(response)

response_json_xy = False
if response_json_xy:
        # Find the first '{' to skip any log lines before the JSON
    match = re.search(r'(\{.*\}|\[.*\])', response, re.DOTALL)

    response_json = match.group(0)
    data = json.loads(response_json)


    # Handle both dict and list outputs
    coords_list = extract_xy_coords(data)


    # Extract x and y as lists
    x_model = [coord["x"] for coord in coords_list]
    y_model = [coord["y"] for coord in coords_list]


    # Ensure both are lists for downstream code
    def ensure_list(val):
        if isinstance(val, (int, float)):
            return [val]
        return list(val)
    x_model = ensure_list(x_model)
    y_model = ensure_list(y_model)
    x_orig = [x * w_orig / w_model for x in x_model]
    y_orig = [y * h_orig / h_model for y in y_model]
    points_orig = list(zip(x_orig, y_orig))
    plot_points_on_image("/home/mani/Downloads/me_resized.jpg", points_orig, point_color='red', point_size=50)
    # points=list(zip([510,420], [420,0.42]))
    # plot_points_on_image("/home/mani/Downloads/me_resized.jpg", points, point_color='red', point_size=50)