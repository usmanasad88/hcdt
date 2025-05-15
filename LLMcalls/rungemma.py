import subprocess
from PIL import Image
import matplotlib.pyplot as plt
import json
import re


import os
import subprocess

def run_llama_mtmd(prompt: str, image_path: str) -> str:
    cmd = [
        "/home/mani/Repos/llama.cpp/build/bin/llama-mtmd-cli",
        "-m", "/home/mani/Repos/llama.cpp/models/gemma-3-12b-it-q4_0.gguf",
        "--mmproj", "/home/mani/Repos/llama.cpp/models/mmproj-model-f16-12B.gguf",
        "-p", prompt,
        "--image", image_path
    ]
    # Copy current environment and remove LD_LIBRARY_PATH
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        output = result.stdout.strip()
        print(output)
        return output
    except subprocess.CalledProcessError as e:
        print("Error during inference:")
        print(e.stderr)
        return None

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

# resize_image_to_model_size("/home/mani/Central/Cooking1/Stack/output_frames/cam1-frame-1.4.jpg", 896, 896, "/home/mani/Downloads/me_resized.jpg")

def run_base_system_prompt():
    base_system_prompt_content = """<start_of_turn>user
    Your Role: You are an AI assistant specialized in real-time human action analysis and state tracking from video frames.
    Your Task: You will be provided with a sequence of video frames. For each frame in the sequence, you must:
    Analyze the frame in the context of the overall task and recent frames (if available).
    Determine the current state of the task.
    Identify the key objects visible in the frame.
    Anticipate the human's immediate next micro-action or transition.
    Conditionally, if you can confidently predict the human's more substantial action and target location over the next approximate 1-2 seconds, provide that prediction.
    Use the provided Task Context Description below as your knowledge base regarding the overall goal, valid states, possible actions, key objects, and environmental layout.
    Input
    Video Data: A sequence of individual video frames, provided sequentially. You should process each frame as it arrives, potentially using information from previously processed frames in the sequence to inform your current analysis.
    Required Output Format (For EACH Frame)
    Provide your analysis for the current frame in a structured format.
    Task State: An update of the task state based on your interpretation of the current frame, including toolbox_placed_on_table, num_chairs_stacked, operator_holding, gaze target, current target object, current phase.
    Identified Key Objects: List the key objects (defined in context) clearly visible in the current frame and their immediate status or location relative to the operator or environment.
    Example: "Operator near Initial Object Area; Toolbox visible on floor; Red Chair 1 visible nearby; Folding Table visible in foreground."

    Expected Immediate Next Action: Describe the most likely very short-term action transition expected to occur immediately following the current frame, based on the current posture, movement dynamics, and task state. Use terms from the Action Decomposition where applicable, but focus on the transition.
    Example: "continue walking towards toolbox"
    Example: "initiate reach for chair handle"
    Example: "complete placing object"
    Example: "stabilize after placing chair"
    Example: "turn body towards table"

    Use the Output structure defined in the context below.
    Example:
    {
    "time": ,
    "toolbox_placed_on_table": ,
    "num_chairs_stacked": ,
    "operator_holding": 
    "gaze_target": ,
    "current_target_object": ,
    "current_phase": ,
    "Identified Key Objects": ,
    "Expected Immediate Next Action": 
    }
    —
    ## Task Context Description ##
    ### 1. Overall Goal:
    The operator's overall goal is to organize the room by stacking several plastic chairs in a designated location against the wall and placing a toolbox onto a small folding table.
    ### 2. Task State Representation:
    A simple state representation can track the status of the key objects:

    ### 3. Action Decomposition:
    The task can be broken down into the following observable, discrete actions:
    Stand idle
    Walk to location (e.g., walk towards toolbox, walk towards chair, walk towards table, walk towards stacking area)
    Reach for object (targeting either the toolbox or a chair)
    Pick up object (lifting the toolbox or a chair)
    Transport object (carrying the held object to a destination)
    Place object (setting the toolbox on the table, placing a chair on the floor/stack)
    Turn towards location/object (Orienting body before walking or interacting)


    ### 4. Action Dependencies:
    General Sequence: Actions often follow a Walk -> Reach -> Pick up -> Transport -> Place pattern for each object being moved.
    Prerequisites:
    Reach for object requires the operator to be near the target object (often preceded by Walk to location or Turn towards object).
    Pick up object requires a preceding Reach for object.
    Transport object requires a preceding Pick up object.
    Place object requires a preceding Transport object.


    Task Order: The operator first moves the toolbox to the table before starting to move the chairs.
    Stacking Logic: Chairs are placed sequentially at the Stacking Location. The first chair is placed on the floor, subsequent chairs are placed on top of the previously placed one.
    Transitions: After Place object, the operator will typically either Stand idle briefly or initiate a Walk to location to retrieve the next object.
    ### 5. Key Objects and Locations (Optional but Recommended):
    Key Objects:
    Operator: The single human performing the task.
    Toolbox: An orange and black case, initially near the chairs.
    Turquoise Chair: One plastic chair of this color.
    Red Chair: Three plastic chairs of this color.
    Folding Table: A small, wooden-topped table in the foreground.


    Key Locations (Conceptual):
    Initial Object Area: The general space where the chairs and toolbox are located at the start (mid-ground, right side).
    Table Location: The position of the folding table (foreground, left-center).
    Stacking Location: The designated area against the far wall, near the refrigerator, where chairs are stacked.


    ### 6. Other Relevant Context:
    Environment: Indoor room with tiled floor, multiple doorways, a refrigerator against the far wall. Space is generally open enough for easy movement.
    Object Properties: Chairs appear lightweight and are standard stackable plastic chairs. The toolbox is carried with two hands.
    Repetition: The core task involves repetitive cycles of picking up, transporting, and placing chairs.
    Implicit Goal: The actions suggest a goal of tidying or organizing the space.
    No Tools: No external tools are used; the task relies solely on the operator's manual actions."""

    MAX_HISTORY_FOR_PROMPT = 3  # Number of past model responses to include in the prompt

    all_responses_log_filepath = "LLMcalls/responsehistory.txt"
    evolving_prompt_filepath = "LLMcalls/prompt.txt"

    # In-memory list to store the last MAX_HISTORY_FOR_PROMPT formatted model responses
    model_responses_for_prompt_construction = []

    # Clear the response log file at the beginning of a full run
    with open(all_responses_log_filepath, "w") as f_log:
        f_log.write("--- Log Start ---\n\n")
        
    for timestamp in range(0, 42): # Loop for 42 images (0 to 41)
        image_path = f"/home/mani/Central/Cooking1/Stack/output_frames/second/cam2_cr-frame-{timestamp}.0.jpg"
        # 1. Construct the prompt to send to the LLM for the current image
        prompt_parts_for_llm = [base_system_prompt_content]
        prompt_parts_for_llm.extend(model_responses_for_prompt_construction) # Add history of model responses

        # Add the user turn for the *current* image
        if timestamp == 0:
            current_user_turn = f"<start_of_turn>user\nHere is the initial image (frame {timestamp}):\n<end_of_turn>"
        else:
            current_user_turn = f"<start_of_turn>user\nThe next image is frame {timestamp}:\n<end_of_turn>"
        prompt_parts_for_llm.append(current_user_turn)
        
        prompt_to_send_to_llm = "\n".join(prompt_parts_for_llm)

        # 2. Call the LLM
        response = run_llama_mtmd(
            prompt=prompt_to_send_to_llm,
            image_path=image_path
        )

        # 3. Log the attempt and response (if any) to responsehistory.txt
        with open(all_responses_log_filepath, "a") as f_log:
            f_log.write(f"--- Image: {os.path.basename(image_path)} (Timestamp: {timestamp}) ---\n")
            f_log.write(f"Prompt sent to LLM:\n{prompt_to_send_to_llm}\n---\n")
            if response:
                f_log.write(f"Model Response:\n{response}\n\n")
            else:
                f_log.write("No response from model.\n\n")

        if response:
            # 4. Add the successful response to our in-memory history for future prompts
            formatted_model_response = f"<start_of_turn>model\n{response}<end_of_turn>"
            model_responses_for_prompt_construction.append(formatted_model_response)

            # 5. Trim history to keep only the last MAX_HISTORY_FOR_PROMPT responses
            if len(model_responses_for_prompt_construction) > MAX_HISTORY_FOR_PROMPT:
                model_responses_for_prompt_construction = model_responses_for_prompt_construction[-MAX_HISTORY_FOR_PROMPT:]
        # If there was no response, model_responses_for_prompt_construction is not updated with this failure,
        # so the next prompt will use the history from before the failure.

        # 6. Construct and save the content for `prompt.txt`
        # This file will contain: base_system_prompt + current_model_history + user_turn_for_NEXT_image
        prompt_parts_for_file = [base_system_prompt_content]
        prompt_parts_for_file.extend(model_responses_for_prompt_construction) # Uses (potentially updated) history

        if timestamp < 41: # If there is a next image in the loop (0-40, so next is 1-41)
            next_user_turn = f"<start_of_turn>user\nThe next image is frame {timestamp+1}:\n<end_of_turn>"
            prompt_parts_for_file.append(next_user_turn)
        # If it's the last image (timestamp 41), no "next user turn" is added to the file.
        
        content_for_evolving_prompt_file = "\n".join(prompt_parts_for_file)
        
        with open(evolving_prompt_filepath, "w") as f_prompt_file: # Overwrite prompt.txt
            f_prompt_file.write(content_for_evolving_prompt_file)


# with open("/home/mani/CLoSD/closd/IntentNet/prompt.txt", "r") as f:
#     originalprompt = f.read()

# prompt=originalprompt
# for timestamp in range(0, 42):
#     image_path = f"/home/mani/Central/Cooking1/Stack/output_frames/second/cam2_cr-frame-{timestamp}.0.jpg"
#     response1 = run_llama_mtmd(
#         prompt=prompt,
#         image_path=image_path
#     )
#     with open("/home/mani/CLoSD/closd/IntentNet/prompt.txt", "a") as f:        
#     with open("/home/mani/CLoSD/closd/IntentNet/prompt.txt", "a") as f:
#         f.write(f"<start_of_turn>model\n{response}<end_of_turn>\n")
#         f.write(f"<start_of_turn>user\nThe next image at time {timestamp+1}<end_of_turn>\n")
    
# def plot_points_on_image(image_path, points, point_color='red', point_size=50):
#     """
#     Plots given (x, y) points on the image.

#     Args:
#         image_path (str): Path to the image file.
#         points (list of tuple): List of (x, y) coordinates.
#         point_color (str): Color of the points.
#         point_size (int): Size of the points.
#     """
#     img = Image.open(image_path)
#     plt.imshow(img)
#     xs, ys = zip(*points)
#     plt.scatter(xs, ys, c=point_color, s=point_size)
#     plt.axis('off')
#     plt.show()

# # # Example usage:
# # w_model = 896
# # h_model = 896
# # with Image.open("/home/mani/Downloads/me_resized.jpg") as img:
# #     w_orig, h_orig = img.size

# def extract_xy_coords(data):
    # """
    # Recursively extract all dicts with 'x' and 'y' keys from data.
    # Returns a list of dicts with 'x' and 'y'.
    # """
    # coords = []
    # if isinstance(data, dict):
    #     # If this dict has x and y, add it
    #     if "x" in data and "y" in data:
    #         coords.append(data)
    #     # Otherwise, search its values
    #     for v in data.values():
    #         coords.extend(extract_xy_coords(v))
    # elif isinstance(data, list):
    #     for item in data:
    #         coords.extend(extract_xy_coords(item))
    # return coords

# if response:
#     print("Model output:")
#     print(response)

# # # Find the first '{' to skip any log lines before the JSON
# # match = re.search(r'(\{.*\}|\[.*\])', response, re.DOTALL)

# # response_json = match.group(0)
# # data = json.loads(response_json)


# # # Handle both dict and list outputs
# # coords_list = extract_xy_coords(data)


# # # Extract x and y as lists
# # x_model = [coord["x"] for coord in coords_list]
# # y_model = [coord["y"] for coord in coords_list]


# # # Ensure both are lists for downstream code
# # def ensure_list(val):
# #     if isinstance(val, (int, float)):
# #         return [val]
# #     return list(val)
# # x_model = ensure_list(x_model)
# # y_model = ensure_list(y_model)
# # x_orig = [x * w_orig / w_model for x in x_model]
# # y_orig = [y * h_orig / h_model for y in y_model]
# # points_orig = list(zip(x_orig, y_orig))

# #plot_points_on_image("/home/mani/Downloads/me_resized.jpg", points_orig, point_color='red', point_size=50)
# points=list(zip([510,420], [420,0.42]))
# plot_points_on_image("/home/mani/Downloads/me_resized.jpg", points, point_color='red', point_size=50)