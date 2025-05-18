import os
import mimetypes
import json # Added import
from google import genai
from google.genai import types

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

# --- End DAG Helper Functions ---

def save_binary_file(file_name, data):
    with open(file_name, "wb") as f:
        f.write(data)
    print(f"File saved to to: {file_name}")

# generate_history function is removed as state is managed directly via JSON.

def generate(prompt_text_with_state, image_path, turn_history=None):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    uploaded_image = client.files.upload(file=image_path)
    # Using a model that is good with JSON and text generation.
    # Consider "gemini-1.5-flash-latest" or "gemini-1.5-pro-latest" if available and suitable.
    # "gemini-2.0-flash-preview-image-generation" might be an older or specialized name.
    # For now, keeping the user's model, but this is a key point for JSON reliability.
    model = "gemini-2.0-flash-preview-image-generation" # Changed to a generally available model good with JSON

    system_prompt = """Your Role: You are an AI assistant specialized in real-time human action analysis and state tracking for a cooking task, using a predefined Directed Acyclic Graph (DAG) of steps.
Your Task: You will be provided with the current state of the cooking task (as a JSON object, including the status of all `dag_steps` and a list of `available_next_dag_steps`), a video frame with both egocentric and exocentric views, and conversation history. For each frame:
1. Analyze the frame in conjunction with the provided current state (especially `dag_steps` statuses and `available_next_dag_steps`) and history.
2. Determine the overall current state of the cooking task (e.g., water boiling, vegetables chopped).
3. Identify key objects visible and the operator's actions.
4. Crucially, determine if the `last_observed_action` corresponds to the completion of one of the `available_next_dag_steps`. If so, specify its ID in `identified_completed_dag_step_id`. If no DAG step is completed, this should be null or an empty string.
5. Output a single JSON object that represents the *updated* state of the task, including your observations, predictions, and any identified completed DAG step.

Task Context Description (Noodle Preparation):
The overall goal is to prepare a noodle dish. The `dag_steps` in the input JSON outlines the specific sequence and dependencies.
Key objects: pot, pan/skillet, stove, knife, cutting board, noodles, cabbage, spring onions, garlic, oil, soy sauce, chili powder, salt, plate, strainer/sieve, spoon.
Possible actions: adding ingredients, stirring, chopping, peeling, washing, draining, placing items, adjusting heat, picking up items, discarding waste.

Output Format:
Ensure your entire response is a single, valid JSON object. Do not use markdown like ```json ... ```.
The JSON object should include, but is not limited to, the following fields:
- "current_phase": (string) e.g., "Initial", "PreparingIngredients", "BoilingWater", "ChoppingVegetables", "Sautéing", "CombiningIngredients", "Plating"
- "pot_on_stove": (boolean)
- "water_boiling": (boolean)
- "noodles_cooking": (boolean)
- "vegetables_status": (object) with keys like "cabbage", "spring_onions", "garlic" and values like "uncut", "chopping", "chopped"
- "pan_prepared_with_oil": (boolean)
- "operator_holding": (string) Description of what the operator is holding (e.g., "Knife", "Cabbage", "None").
- "gaze_target": (string) Inferred gaze target.
- "current_target_object": (string) Inferred current target object for interaction.
- "identified_key_objects": (array of strings) List of key objects visible and relevant.
- "last_observed_action": (string) Description of the most recent significant action observed in the current frame.
- "expected_immediate_next_action": (string) Your prediction for the next immediate micro-action.
- "identified_completed_dag_step_id": (string or null) The ID of the DAG step completed by the `last_observed_action`, if applicable from the `available_next_dag_steps`. Otherwise, null or empty string.
Give an example of the expected output."""

    contents = [
        types.Content( # System prompt defining role, task, and output format
            role="user",
            parts=[types.Part.from_text(text=system_prompt)]
        ),
        types.Content( # Example of the model's expected JSON output structure for the cooking task
            role="model",
            parts=[
                types.Part.from_text(text="""{
  "current_phase": "ChoppingVegetables",
  "pot_on_stove": true,
  "water_boiling": true,
  "noodles_cooking": false,
  "vegetables_status": {
    "cabbage": "chopped",
    "spring_onions": "uncut",
    "garlic": "peeling"
  },
  "pan_prepared_with_oil": false,
  "operator_holding": "Garlic Clove",
  "gaze_target": "Garlic on cutting board",
  "current_target_object": "Garlic Clove",
  "identified_key_objects": ["Operator", "Cutting Board", "Knife", "Garlic Cloves", "Pot on stove (background)", "Chopped Cabbage"],
  "last_observed_action": "Operator picked up a garlic clove and is starting to peel it.",
  "expected_immediate_next_action": "Continue peeling garlic clove.",
  "identified_completed_dag_step_id": "step_12"
}""")
            ],
        )
    ]

    if turn_history: # Add previous turns for context
        contents.extend(turn_history)

    # Add the current user turn (image and the prompt containing the current state)
    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_uri(
                    file_uri=uploaded_image.uri,
                    mime_type=uploaded_image.mime_type,
                ),
                types.Part.from_text(text=prompt_text_with_state), # This contains the current state JSON
            ],
        )
    )

    generate_content_config = types.GenerateContentConfig(
        response_modalities=["IMAGE", "TEXT"], # Expecting JSON text
        response_mime_type="text/plain", # The API will return plain text, which we parse as JSON
        # Consider adding temperature if results are too rigid or too creative
        # temperature=0.7 
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

    # try:
    #     response = client.models.generate_content_stream( # Using non-streaming for simpler JSON handling
    #         model=model,
    #         contents=contents,
    #         config=generate_content_config,
    #     )
    #     if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
    #         full_text = response.candidates[0].content.parts[0].text
    #     else:
    #         print("Warning: No valid content in LLM response.")
    #         full_text = "{}" # Return empty JSON on error
            
    # except Exception as e:
    #     print(f"Error during content generation: {e}")
    #     # Return error as JSON string to be handled by the main loop
    #     return json.dumps({"error": "Content generation failed", "details": str(e)})

    # return full_text.strip()


if __name__ == "__main__":
    results_store_file = "LLMcalls/store_results.txt"
    max_history_turns = 3  # Number of previous user/model turns to keep in context

    # Load DAG and initialize state
    initial_dag_steps = load_dag()
    if not initial_dag_steps:
        print("Failed to load DAG. Exiting.")
        exit()

    current_cooking_state = {
        "current_phase": "Initial",
        "pot_on_stove": False,
        "water_boiling": False,
        "noodles_cooking": False,
        "vegetables_status": {
            "cabbage": "uncut",
            "spring_onions": "uncut",
            "garlic": "uncut"
        },
        "pan_prepared_with_oil": False,
        "operator_holding": "None",
        "gaze_target": "None",
        "current_target_object": "None",
        "identified_key_objects": [],
        "last_observed_action": "None",
        "expected_immediate_next_action": "None",
        "dag_steps": initial_dag_steps, # Full DAG with statuses
        "available_next_dag_steps": [], # Will be populated each loop with {id, description}
        "last_completed_dag_step_id": None,
        # "identified_completed_dag_step_id": None # This will come from LLM, not stored directly here but used for update
    }

    conversation_history = [] # Stores types.Content objects for multi-turn context

    with open(results_store_file, "w") as f_out:
        # Reduced loop for testing, adjust as needed
        for second in range(0, 575): # Example: 0 to 4 seconds
            timestamp = f"{second:05d}"
            image_path = f"/home/mani/Central/Cooking1/combined_frames/frame-{timestamp}.jpg"

            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}, skipping.")
                continue

            # 1. Update available DAG steps based on current DAG statuses
            current_cooking_state["available_next_dag_steps"] = get_available_steps(current_cooking_state["dag_steps"])
            
            # 2. Prepare prompt for LLM, including the current state
            # For clarity, we create a temporary dict for the prompt to avoid sending the full dag_steps if it's too verbose,
            # but the system prompt implies the LLM should be aware of all statuses.
            # Let's send a slightly condensed version of dag_steps for the prompt if needed,
            # or ensure the LLM handles the full list well.
            # For now, sending the full current_cooking_state.
            
            # Create a deep copy for the prompt to avoid modifying the main state object if we were to prune it
            state_for_prompt = json.loads(json.dumps(current_cooking_state))
            # Optionally, simplify dag_steps in state_for_prompt if too large:
            # state_for_prompt["dag_steps"] = [{"id": s["id"], "description": s["description"], "status": s["status"]} for s in state_for_prompt["dag_steps"]]


            current_state_json_str = json.dumps(state_for_prompt, indent=2)
            prompt_for_llm = f"""Current Task State:
            {current_state_json_str}

            Image Timestamp: {timestamp}
            Analyze the provided image and the current task state (especially `dag_steps` statuses and `available_next_dag_steps`).
            Identify if the observed action completes one of the available DAG steps.
            Output the updated task state as a single JSON object. Ensure the output is ONLY the JSON object.
            """
            # Get limited history for context
            context_to_pass = conversation_history[-(max_history_turns*2):] if len(conversation_history) > max_history_turns*2 else conversation_history[:]


            # 3. Call LLM
            llm_response_str = generate(prompt_for_llm, image_path, turn_history=context_to_pass)
            
            llm_output_data = None
            try:
                # Attempt to parse JSON from the response
                llm_output_data = json.loads(llm_response_str)

                if "error" in llm_output_data: # Check for errors from the generate function itself
                        raise Exception(f"LLM generation error: {llm_output_data.get('details', llm_response_str)}")

                # 4. Update current_cooking_state based on the validated llm_output_data
                # Update general state fields (excluding DAG-specific fields handled below)
                for key, value in llm_output_data.items():
                    if key in current_cooking_state and key not in ["dag_steps", "available_next_dag_steps", "last_completed_dag_step_id"]:
                        if isinstance(current_cooking_state[key], dict) and isinstance(value, dict):
                            current_cooking_state[key].update(value) # Merge sub-dictionaries
                        else:
                            current_cooking_state[key] = value
                    # else:
                        # print(f"Warning: LLM returned key '{key}' not in current_cooking_state or handled separately.")
                
                current_cooking_state["last_observed_action"] = llm_output_data.get("last_observed_action", current_cooking_state.get("last_observed_action", "None"))


                # Process identified completed DAG step
                completed_step_id_from_llm = llm_output_data.get("identified_completed_dag_step_id")
                if completed_step_id_from_llm and isinstance(completed_step_id_from_llm, str) and completed_step_id_from_llm.strip():
                    # Verify this step was actually 'available' or 'pending' (get_available_steps marks them 'available')
                    is_valid_completion = False
                    for available_step in current_cooking_state["available_next_dag_steps"]: # Check against the list sent to LLM
                        if available_step["id"] == completed_step_id_from_llm:
                            is_valid_completion = True
                            break
                    
                    if is_valid_completion:
                        # Check current status before updating, to avoid issues if LLM is confused
                        current_status_of_step = ""
                        for step_in_dag in current_cooking_state["dag_steps"]:
                            if step_in_dag["id"] == completed_step_id_from_llm:
                                current_status_of_step = step_in_dag["status"]
                                break
                        
                        if current_status_of_step == 'available': # Only update if it was genuinely available
                            if update_step_status(current_cooking_state["dag_steps"], completed_step_id_from_llm, "completed"):
                                current_cooking_state["last_completed_dag_step_id"] = completed_step_id_from_llm
                                step_desc = get_step_description_by_id(current_cooking_state['dag_steps'], completed_step_id_from_llm)
                                print(f"INFO: LLM identified completion of DAG step: {completed_step_id_from_llm} - '{step_desc}'")
                            else:
                                    print(f"ERROR: Failed to update status for supposedly completed step '{completed_step_id_from_llm}'.")
                        elif current_status_of_step == 'completed':
                            print(f"INFO: LLM re-confirmed already completed step '{completed_step_id_from_llm}'. No status change.")
                        else:
                            print(f"WARNING: LLM identified step '{completed_step_id_from_llm}' as completed, but its current status is '{current_status_of_step}', not 'available'. Check LLM logic or DAG state.")
                    else:
                        print(f"WARNING: LLM identified step '{completed_step_id_from_llm}' as completed, but it was NOT in the 'available_next_dag_steps' list sent to LLM. Ignoring.")
                # else:
                    # print("INFO: LLM did not identify a completed DAG step in this frame.")
                

            except json.JSONDecodeError as e:
                print(f"ERROR: Decoding LLM JSON response failed: {e}")
                print(f"LLM Raw Response: '{llm_response_str}'")
                llm_output_data = {"error": "Failed to parse LLM response", "raw_response": llm_response_str} # For logging
            except Exception as e: # Catch other potential errors during processing
                print(f"ERROR: An unexpected error occurred processing LLM response: {e}")
                print(f"LLM Raw Response: '{llm_response_str}'")
                llm_output_data = {"error": "Unexpected error processing response", "raw_response": llm_response_str} # For logging


            # Add the current user prompt and model's (raw) response to conversation_history
            conversation_history.append(
                types.Content(role="user", parts=[types.Part.from_text(text=prompt_for_llm)])
            )
            # Store the raw string response for history, as it's what the model actually produced
            conversation_history.append(
                types.Content(role="model", parts=[types.Part.from_text(text=llm_response_str)])
            )

            # Keep history to a manageable size
            if len(conversation_history) > max_history_turns * 2:
                conversation_history = conversation_history[-(max_history_turns * 2):]


            print(f"Timestamp: {timestamp}")
            if llm_output_data and "error" not in llm_output_data:
                # print(f"LLM Parsed Output: {json.dumps(llm_output_data, indent=2)}\n")
                print(f"Updated State (summary): Phase: {current_cooking_state['current_phase']}, Last Action: {current_cooking_state['last_observed_action']}, Last DAG Step: {current_cooking_state['last_completed_dag_step_id']}\n")
                f_out.write(f"Timestamp: {timestamp}\n")
                f_out.write(f"LLM Parsed Output: {json.dumps(llm_output_data)}\n") # Log what LLM actually said
                f_out.write(f"Current Full State: {json.dumps(current_cooking_state)}\n\n") # Log full state after update
            else:
                print(f"LLM Processing Error or No Data: {llm_output_data}\n")
                f_out.write(f"Timestamp: {timestamp}\n")
                f_out.write(f"LLM Processing Error or No Data: {json.dumps(llm_output_data)}\n")
                f_out.write(f"Current Full State (before error handling): {json.dumps(current_cooking_state)}\n\n")
            f_out.flush() # Ensure data is written to file immediately