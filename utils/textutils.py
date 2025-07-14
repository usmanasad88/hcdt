import json
import re
from typing import Optional
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.motionutils import get_hand_xy_positions

def parse_yx(response):
    try:
        # Extract the first JSON array in the response
        match = re.search(r'(\[\s*\{.*?\}\s*\])', response, re.DOTALL)
        if not match:
            print("No JSON array found in response.")
            return None, None
        json_str = match.group(1)
        data = json.loads(json_str)
        if isinstance(data, list) and len(data) > 0 and "point" in data[0]:
            y, x = data[0]["point"]
            return y, x
    except Exception as e:
        print(f"Could not parse response or draw point: {e}")
        return None, None
    

input_path = "data/gemini-flash-image-gen-3-memv1.json"
output_path = "data/gemini-flash-image-gen-3-memv1.fixed.json"

def extract_json_from_response(response_string: str) -> Optional[str]:
    """
    Extracts a JSON string from a response that might be wrapped in Markdown code fences,
    triple-double-quotes, or mixed with reasoning text.
    
    Args:
        response_string (str): The raw response string from the model.
        
    Returns:
        Optional[str]: The extracted JSON string, or None if the input is None or empty.
    """
    if not response_string:
        return None

    cleaned_string = response_string.strip()

    # Case 1: Look for JSON within ```json ... ``` blocks
    json_block_match = re.search(r'```json\s*(.*?)\s*```', cleaned_string, re.DOTALL)
    if json_block_match:
        return json_block_match.group(1).strip()
    
    # Case 2: Look for JSON within generic ``` ... ``` blocks
    generic_block_match = re.search(r'```\s*([\{\[].*?[\}\]])\s*```', cleaned_string, re.DOTALL)
    if generic_block_match:
        return generic_block_match.group(1).strip()
    
    # Case 3: Look for JSON within """...""" blocks
    triple_quote_match = re.search(r'"""\s*([\{\[].*?[\}\]])\s*"""', cleaned_string, re.DOTALL)
    if triple_quote_match:
        return triple_quote_match.group(1).strip()
    
    # Case 4: Look for standalone JSON objects/arrays anywhere in the text
    # This will find the last complete JSON object or array in the response
    json_pattern = r'(\{(?:[^{}]|{[^{}]*})*\}|\[(?:[^\[\]]|\[[^\[\]]*\])*\])'
    json_matches = re.findall(json_pattern, cleaned_string, re.DOTALL)
    
    if json_matches:
        # Try to parse each match to ensure it's valid JSON
        for match in reversed(json_matches):  # Start from the last match
            try:
                # Test if it's valid JSON
                json.loads(match)
                return match.strip()
            except json.JSONDecodeError:
                continue
    
    # Case 5: If the entire string starts and ends with { } or [ ], treat it as JSON
    if (cleaned_string.startswith("{") and cleaned_string.endswith("}")) or \
       (cleaned_string.startswith("[") and cleaned_string.endswith("]")):
        return cleaned_string

    # If no JSON is found, return None
    return None

def fix_json_file(input_path, output_path):
    data = []
    current_json_lines = []
    with open(input_path, "r") as f:
        for line in f:
            line = line.rstrip()
            # If line starts with a number and a tab, it's a new record
            if re.match(r'^\d+(\.\d+)?\s*\t', line):
                # If we have accumulated JSON lines, process them
                if current_json_lines:
                    json_str = "\n".join(current_json_lines).strip()
                    try:
                        obj = json.loads(json_str)
                        data.append(obj)
                    except Exception as e:
                        print(f"Error parsing JSON:\n{json_str}\n{e}")
                    current_json_lines = []
                # Start new JSON part (after the tab)
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    current_json_lines = [parts[1].strip()]
            else:
                # Continuation of previous JSON
                if line.strip():  # skip empty lines
                    current_json_lines.append(line.strip())
        # Don't forget the last one
        if current_json_lines:
            json_str = "\n".join(current_json_lines).strip()
            try:
                obj = json.loads(json_str)
                data.append(obj)
            except Exception as e:
                print(f"Error parsing JSON:\n{json_str}\n{e}")

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

def log_to_json(input_path, output_path):
    """
    Convert a log file to a JSON file.
    The log file is expected to have lines starting with "Timestamp:", "LLM Parsed Output:", and "Current Full State:".
    """
    # Read the log file and parse it into a list of dictionaries
    records = []
    with open(input_path, "r") as f:
        block = {}
        for line in f:
            line = line.strip()
            if line.startswith("Timestamp:"):
                if block:
                    records.append(block)
                    block = {}
                block["timestamp"] = line.split(":", 1)[1].strip()
            elif line.startswith("LLM Parsed Output:"):
                json_str = line.split(":", 1)[1].strip()
                try:
                    block["llm_parsed_output"] = json.loads(json_str)
                except Exception as e:
                    block["llm_parsed_output"] = json_str  # fallback
            elif line.startswith("Current Full State:"):
                json_str = line.split(":", 1)[1].strip()
                try:
                    block["current_full_state"] = json.loads(json_str)
                except Exception as e:
                    block["current_full_state"] = json_str  # fallback
        if block:
            records.append(block)

    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)

def get_action_description_for_frame(frame_number: int, gt_file_path: str, dag_file_path: str) -> Optional[str]:
    """
    Get the action description from dag.json corresponding to the steps_in_progress 
    from the ground truth file for a given frame number.
    
    Args:
        frame_number (int): The frame number to look up
        gt_file_path (str): Path to the ground truth JSON file (e.g., S02A08I21_gt.json)
        dag_file_path (str): Path to the DAG JSON file (dag.json)
        
    Returns:
        Optional[str]: The action description if found, None otherwise
    """
    try:
        # Load ground truth data
        with open(gt_file_path, 'r') as f:
            gt_data = json.load(f)
        
        # Load DAG data
        with open(dag_file_path, 'r') as f:
            dag_data = json.load(f)
        
        # Create a mapping from step ID to description, handling both formats
        step_descriptions = {}
        for step in dag_data:
            step_id = step.get('id')
            description = step.get('description') or step.get('action', 'Unknown')
            
            # Handle both string and numeric IDs
            step_descriptions[step_id] = description
            step_descriptions[str(step_id)] = description  # Also store as string
            if isinstance(step_id, str) and step_id.isdigit():
                step_descriptions[int(step_id)] = description  # Also store as int if possible
        
        # Find the frame data
        frame_data = None
        for entry in gt_data:
            entry_frame = entry.get('frame') or entry.get('frame_number')
            if entry_frame == frame_number:
                frame_data = entry
                break
        
        if not frame_data:
            print(f"Frame {frame_number} not found in ground truth data")
            return None
        
        # Get steps in progress
        steps_in_progress = frame_data.get('steps_in_progress', [])
        
        if not steps_in_progress:
            # Get available steps when no action is in progress
            steps_available = frame_data.get('steps_available', [])
            
            if not steps_available:
                return "No action in progress and no steps available"
            
            # Get descriptions for available steps
            available_descriptions = []
            for step_id in steps_available:
                # Check if it's already a description or needs to be looked up
                if step_id in step_descriptions:
                    available_descriptions.append(step_descriptions[step_id])
                else:
                    # If not found in mapping, it might already be a description
                    available_descriptions.append(str(step_id))
                    print(f"Warning: Step ID '{step_id}' not found in DAG, using as-is")
            
            if available_descriptions:
                return f"No action in progress. Available steps: {'; '.join(available_descriptions)}"
            else:
                return "No action in progress and no valid available steps found"
        
        # Get descriptions for all steps in progress
        descriptions = []
        for step_id in steps_in_progress:
            # Check if it's already a description or needs to be looked up
            if step_id in step_descriptions:
                descriptions.append(step_descriptions[step_id])
            else:
                # If not found in mapping, it might already be a description
                descriptions.append(str(step_id))
                print(f"Warning: Step ID '{step_id}' not found in DAG, using as-is")
        
        # Join multiple descriptions if there are multiple steps in progress
        if descriptions:
            return "; ".join(descriptions)
        else:
            return "Unknown action"
            
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None
    except Exception as e:
        print(f"Error getting action description: {e}")
        return None


def get_all_actions_for_frame(frame_number: int, gt_file_path: str, dag_file_path: str) -> dict:
    """
    Get comprehensive action information for a given frame including completed, 
    in progress, and available steps with their descriptions.
    
    Args:
        frame_number (int): The frame number to look up
        gt_file_path (str): Path to the ground truth JSON file
        dag_file_path (str): Path to the DAG JSON file
        
    Returns:
        dict: Dictionary containing all action information for the frame
    """
    try:
        # Load ground truth data
        with open(gt_file_path, 'r') as f:
            gt_data = json.load(f)
        
        # Load DAG data
        with open(dag_file_path, 'r') as f:
            dag_data = json.load(f)
        
        # Create a mapping from step ID to description, handling both formats
        step_descriptions = {}
        for step in dag_data:
            step_id = step.get('id')
            description = step.get('description') or step.get('action', 'Unknown')
            
            # Handle both string and numeric IDs
            step_descriptions[step_id] = description
            step_descriptions[str(step_id)] = description  # Also store as string
            if isinstance(step_id, str) and step_id.isdigit():
                step_descriptions[int(step_id)] = description  # Also store as int if possible
        
        # Find the frame data
        frame_data = None
        actual_frame_number = None
        for entry in gt_data:
            entry_frame = entry.get('frame') or entry.get('frame_number')
            if entry_frame == frame_number:
                frame_data = entry
                actual_frame_number = entry_frame
                break
        
        if not frame_data:
            return {"error": f"Frame {frame_number} not found in ground truth data"}
        
        # Helper function to get description for a step
        def get_step_description(step_id):
            if step_id in step_descriptions:
                return step_descriptions[step_id]
            else:
                # If not found in mapping, it might already be a description
                return str(step_id)
        
        # Build comprehensive action information
        result = {
            "frame": actual_frame_number,
            "operator_holding": frame_data.get('operator_holding', 'unknown'),
            "gaze_target": frame_data.get('gaze_target', 'unknown'),
            "steps_completed": [
                {"id": step_id, "description": get_step_description(step_id)}
                for step_id in frame_data.get('steps_completed', [])
            ],
            "steps_in_progress": [
                {"id": step_id, "description": get_step_description(step_id)}
                for step_id in frame_data.get('steps_in_progress', [])
            ],
            "steps_available": [
                {"id": step_id, "description": get_step_description(step_id)}
                for step_id in frame_data.get('steps_available', [])
            ]
        }
        
        # Add current action description
        if frame_data.get('steps_in_progress'):
            current_actions = [get_step_description(step_id) for step_id in frame_data['steps_in_progress']]
            result["current_action"] = "; ".join(current_actions)
        else:
            result["current_action"] = "No action in progress"
        
        return result
        
    except Exception as e:
        return {"error": f"Error getting action information: {e}"}

def compare_phase2_results(input_path: str, output_path: str):
    """
    Compare the Phase 2 results by extracting only the relevant JSON data, and comparing with results from pt
    
    Args:
        input_path (str): Path to the input JSON file containing Phase 2 results.
        output_path (str): Path to save the simplified JSON data.
    """

    vitpose_file = "/home/mani/Central/HaVid/S02A08I21/GVHMR/front/preprocess/vitpose.pt"

    # Load the phase 2 results
    with open(input_path, 'r') as f:
        phase2_data = json.load(f)
    
    # Extract simplified results
    simplified_results = []
    
    for entry in phase2_data:
        frame_number = entry["prediction_frame"]
        left_hand_x, left_hand_y, right_hand_x, right_hand_y = get_hand_xy_positions(
                            vitpose_file, frame_number=frame_number)
        
        simplified_entry = {
            "frame": entry["prediction_frame"],
            "predicted_hand_positions": {
                "left_hand_x": entry["predicted_positions"]["left_hand_x"],
                "left_hand_y": entry["predicted_positions"]["left_hand_y"],
                "right_hand_x": entry["predicted_positions"]["right_hand_x"],
                "right_hand_y": entry["predicted_positions"]["right_hand_y"]
            },
            "actual_hand_positions": {
                "left_hand_x": left_hand_x,
                "left_hand_y": left_hand_y,
                "right_hand_x": right_hand_x,
                "right_hand_y": right_hand_y    
            },
        }
        simplified_results.append(simplified_entry)
    
    # Save the simplified results
    with open(output_path, 'w') as f:
        json.dump(simplified_results, f, indent=2)
    
    print(f"✅ Simplified Phase 2 results saved to {output_path}")
    print(f"📊 Processed {len(simplified_results)} prediction frames")


def simplify_phase2_results(input_path: str, output_path: str):
    """
    Simplify the Phase 2 results by extracting only the relevant JSON data.
    
    Args:
        input_path (str): Path to the input JSON file containing Phase 2 results.
        output_path (str): Path to save the simplified JSON data.
    """
    # Load the phase 2 results
    with open(input_path, 'r') as f:
        phase2_data = json.load(f)
    
    # Extract simplified results
    simplified_results = []
    
    for entry in phase2_data:
        simplified_entry = {
            "frame": entry["prediction_frame"],
            "predicted_positions": {
                "left_hand_x": entry["predicted_positions"]["left_hand_x"],
                "left_hand_y": entry["predicted_positions"]["left_hand_y"],
                "right_hand_x": entry["predicted_positions"]["right_hand_x"],
                "right_hand_y": entry["predicted_positions"]["right_hand_y"]
            }
        }
        simplified_results.append(simplified_entry)
    
    # Save the simplified results
    with open(output_path, 'w') as f:
        json.dump(simplified_results, f, indent=2)
    
    print(f"✅ Simplified Phase 2 results saved to {output_path}")
    print(f"📊 Processed {len(simplified_results)} prediction frames")

 

# Example usage:
if __name__ == "__main__":
    input_file = "data/HAViD/phase2_result.json"
    output_file = "data/HAViD/phase2_simplified.json"
    
    compare_phase2_results(input_file, output_file)
#     gt_path = "data/HAViD/S02A08I21_gt.json"
#     dag_path = "data/HAViD/dag.json"
    
#     # Test the function
#     frame = 1711
#     action = get_action_description_for_frame(frame, gt_path, dag_path)
#     print(f"Frame {frame}: {action}")
    
#     # Get comprehensive information
#     # all_info = get_all_actions_for_frame(frame, gt_path, dag_path)
#     # print(f"\nComprehensive info for frame {frame}:")
#     # print(json.dumps(all_info, indent=2))