import json
import re
from typing import Optional, Dict
import sys
import os
from graphviz import Digraph

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
    

# input_path = "data/gemini-flash-image-gen-3-memv1.json"
# output_path = "data/gemini-flash-image-gen-3-memv1.fixed.json"

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

    def convert_python_booleans(json_str: str) -> str:
        """Convert Python boolean strings to JSON format"""
        # Replace standalone True/False with true/false
        json_str = re.sub(r'\bTrue\b', 'true', json_str)
        json_str = re.sub(r'\bFalse\b', 'false', json_str)
        return json_str

    # Case 1: Look for JSON within ```json ... ``` blocks
    json_block_match = re.search(r'```json\s*(.*?)\s*```', cleaned_string, re.DOTALL)
    if json_block_match:
        return convert_python_booleans(json_block_match.group(1).strip())
    
    # Case 2: Look for JSON within generic ``` ... ``` blocks
    generic_block_match = re.search(r'```\s*([\{\[].*?[\}\]])\s*```', cleaned_string, re.DOTALL)
    if generic_block_match:
        return convert_python_booleans(generic_block_match.group(1).strip())
    
    # Case 3: Look for JSON within """...""" blocks
    triple_quote_match = re.search(r'"""\s*([\{\[].*?[\}\]])\s*"""', cleaned_string, re.DOTALL)
    if triple_quote_match:
        return convert_python_booleans(triple_quote_match.group(1).strip())
    
    # Case 4: Look for standalone JSON objects/arrays anywhere in the text
    # This will find the last complete JSON object or array in the response
    json_pattern = r'(\{(?:[^{}]|{[^{}]*})*\}|\[(?:[^\[\]]|\[[^\[\]]*\])*\])'
    json_matches = re.findall(json_pattern, cleaned_string, re.DOTALL)
    
    if json_matches:
        # Try to parse each match to ensure it's valid JSON
        for match in reversed(json_matches):  # Start from the last match
            try:
                # Convert booleans first, then test if it's valid JSON
                converted_match = convert_python_booleans(match)
                json.loads(converted_match)
                return converted_match.strip()
            except json.JSONDecodeError:
                continue
    
    # Case 5: If the entire string starts and ends with { } or [ ], treat it as JSON
    if (cleaned_string.startswith("{") and cleaned_string.endswith("}")) or \
       (cleaned_string.startswith("[") and cleaned_string.endswith("]")):
        return convert_python_booleans(cleaned_string)

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
        
        # Sort the data by frame number to ensure proper ordering
        gt_data.sort(key=lambda x: x.get('frame') or x.get('frame_number') or 0)
        
        # Find the appropriate ground truth entry
        # The ground truth for a frame is the last entry with frame_number <= target frame
        frame_data = None
        for entry in gt_data:
            entry_frame = entry.get('frame') or entry.get('frame_number')
            if entry_frame is None:
                continue
                
            if entry_frame <= frame_number:
                frame_data = entry
            else:
                # Since data is sorted, we can break once we exceed the target frame
                break
        
        if not frame_data:
            print(f"Frame {frame_number} not found in ground truth data")
            return None
        
        # Get steps in progress

        steps_in_progress = frame_data['state'].get('steps_in_progress', [])
        
        if not steps_in_progress:
            # Get available steps when no action is in progress
            steps_available = frame_data['state'].get('steps_available', [])

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

def get_keystep_for_frame(frame_number: int, gt_file_path: str) -> Optional[str]:
    """
    Get the action description for a given frame number by checking which action
    is in progress (frame number falls between start_frame and end_frame).
    If no action is in progress, returns the next upcoming action.
    
    Args:
        frame_number (int): The frame number to look up
        gt_file_path (str): Path to the ground truth JSON file containing actions
        
    Returns:
        Optional[str]: The action description if found, or next action info if none in progress, None on error
    """
    try:
        # Load ground truth data
        with open(gt_file_path, 'r') as f:
            gt_data = json.load(f)
        
        # Handle the case where data has an "actions" key containing the array
        actions_list = gt_data.get("actions", gt_data) if isinstance(gt_data, dict) else gt_data
        
        # Find the action that contains this frame number
        for action in actions_list:
            start_frame = action.get('start_frame')
            end_frame = action.get('end_frame')
            
            if start_frame is not None and end_frame is not None:
                if start_frame <= frame_number <= end_frame:
                    return action.get('action', 'Unknown action')
        
        # If no action found for this frame, find the next upcoming action
        next_action = None
        min_start_frame = float('inf')
        
        for action in actions_list:
            start_frame = action.get('start_frame')
            if start_frame is not None and start_frame > frame_number:
                if start_frame < min_start_frame:
                    min_start_frame = start_frame
                    next_action = action.get('action', 'Unknown action')
        
        if next_action:
            return f"No keystep at current. The next keystep is {next_action}"
        else:
            return f"No keystep at current. No upcoming keysteps found after frame {frame_number}"
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None
    except Exception as e:
        print(f"Error getting keystep for frame: {e}")
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
        
        # Sort the data by frame number to ensure proper ordering
        gt_data.sort(key=lambda x: x.get('frame') or x.get('frame_number') or 0)
        
        # Find the appropriate ground truth entry
        # The ground truth for a frame is the last entry with frame_number <= target frame
        frame_data = None
        actual_frame_number = None
        for entry in gt_data:
            entry_frame = entry.get('frame') or entry.get('frame_number')
            if entry_frame is None:
                continue
                
            if entry_frame <= frame_number:
                frame_data = entry
                actual_frame_number = entry_frame
            else:
                # Since data is sorted, we can break once we exceed the target frame
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

def set_frame_number_in_json(input_json_path, fps, output_path=None):
    """
    Set the 'frame' or 'frame_number' key in the JSON data according to the time, or timestamp key, for the given fps.
    
    Args:
        input_json_path (string): path of the input JSON file.
        fps (int): frame number.
        output_path (string, optional): path to save the modified JSON file. If None, modifies in place.
        
    Returns:
        Creates output path, returns nothing.
    """
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    alt=True
    if alt:
        # Handle the case where data has an "actions" key containing the array
        actions_list = data.get("actions", data) if isinstance(data, dict) else data
        
        for entry in actions_list:
            start_time=entry['start_time'] # MM:SS
            start_seconds= int(start_time.split(':')[0]) * 60 + int(start_time.split(':')[1])
            start_frame = int(start_seconds * fps)
            entry['start_frame'] = start_frame

            end_time=entry['end_time']
            end_seconds= int(end_time.split(':')[0]) * 60 + int(end_time.split(':')[1])
            end_frame = int(end_seconds * fps)
            entry['end_frame'] = end_frame

    conventional = False
    if conventional:
        for entry in data:
            if 'time' in entry:
                # Calculate frame number based on time and fps
                # If time is say 1.15, it means 1 minutes and 15 seconds, so we convert it to total seconds
                minutes, seconds = divmod(entry['time'], 1)
                total_seconds = int(minutes * 60 + seconds * 100)
                entry['frame_number'] = int(total_seconds * fps)
            elif 'timestamp' in entry:
                # Calculate frame number based on timestamp and fps
                entry['frame'] = int(entry['timestamp'] * fps)
            else:
                print(f"Warning: No 'time' or 'timestamp' key found in entry: {entry}")
        
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Modified JSON saved to {output_path}")
    else:
        with open(input_json_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Modified JSON saved to {input_json_path}")

# # Example usage:
# if __name__ == "__main__":
#     set_frame_number_in_json("data/Cooking/fair_cooking_05_02_gt_alt.json", 30)
    # input_file = "data/HAViD/phase2_result.json"
    # output_file = "data/HAViD/phase2_simplified.json"
    
    # compare_phase2_results(input_file, output_file)
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



def create_task_graph_visualization(json_data_string: str, output_filename: str = 'task_graph'):
    """
    Takes a JSON string representing a task action graph, creates a Graphviz
    visualization, and saves it as a DOT file and a PNG image.

    Args:
        json_data_string (str): A JSON string containing the task actions
                                with 'id', 'action', and 'dependencies'.
        output_filename (str): The base name for the output files (e.g.,
                                'task_graph' will generate 'task_graph.dot'
                                and 'task_graph.png').
    """
    try:
        # 1. Parse the JSON input string
        task_actions = json.loads(json_data_string)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return

    # 2. Create a Digraph object (for directed graph)
    #    'comment' is for internal Graphviz use, 'strict' avoids duplicate edges
    dot = Digraph(comment='Task Action Graph', strict=True)

    # Set some global graph attributes for better appearance
    dot.graph_attr['rankdir'] = 'LR' # Layout from Left to Right (alternatively 'TB' for Top to Bottom)
    dot.graph_attr['overlap'] = 'false' # Avoid node overlaps
    dot.graph_attr['splines'] = 'true' # Smooth edges

    # Set some global node attributes
    dot.node_attr['shape'] = 'box' # Nodes will be rectangular boxes
    dot.node_attr['style'] = 'rounded,filled' # Rounded corners and filled
    dot.node_attr['fillcolor'] = '#ADD8E6' # Light blue background for nodes
    dot.node_attr['fontname'] = 'Helvetica' # Use a common sans-serif font

    # Set some global edge attributes
    dot.edge_attr['color'] = '#696969' # Dark grey edges
    dot.edge_attr['arrowhead'] = 'vee' # V-shaped arrowheads
    dot.edge_attr['penwidth'] = '1.5' # Thicker lines for edges

    # Create a dictionary to easily map IDs to action names for labels
    action_labels = {action['id']: action['action'] for action in task_actions}

    # 3. Add nodes
    for action in task_actions:
        node_id = str(action['id']) # Graphviz expects string IDs
        node_label = action['action']
        dot.node(node_id, node_label)

    # 4. Add edges based on dependencies
    for action in task_actions:
        current_id = str(action['id'])
        for dep_id in action['dependencies']:
            # Ensure the dependency exists as a node before adding an edge
            if dep_id in action_labels:
                dot.edge(str(dep_id), current_id)
            else:
                print(f"Warning: Dependency ID {dep_id} for action '{action['action']}' not found in provided actions.")

    # 5. Output the graph
    # Save as DOT file
    # dot.render(output_filename, view=False, format='dot')
    # print(f"Graph DOT file saved as '{output_filename}.dot'")

    # # Save as PNG image
    # # Graphviz automatically looks for the dot executable in your PATH
    # dot.render(output_filename, view=False, format='png')
    # print(f"Graph PNG image saved as '{output_filename}.png'")

    # You can also render to other formats like SVG for scalability
    dot.render(output_filename, view=False, format='svg')
    print(f"Graph SVG image saved as '{output_filename}.svg'")



# --- Example Usage ---
# # Your provided JSON data as a multi-line string
# task_json_input = """
# [
#   {
#     "id": 1,
#     "action": "Walk to toolbox",
#     "dependencies": []
#   },
#   {
#     "id": 2,
#     "action": "Pick up toolbox",
#     "dependencies": [
#       1
#     ]
#   },
#   {
#     "id": 3,
#     "action": "Transport toolbox",
#     "dependencies": [
#       2
#     ]
#   },
#   {
#     "id": 4,
#     "action": "Place toolbox on table",
#     "dependencies": [
#       3
#     ]
#   },
#   {
#     "id": 5,
#     "action": "Walk to turquoise chair",
#     "dependencies": [
#       4
#     ]
#   },
#   {
#     "id": 6,
#     "action": "Pick up turquoise chair",
#     "dependencies": [
#       5
#     ]
#   },
#   {
#     "id": 7,
#     "action": "Transport turquoise chair",
#     "dependencies": [
#       6
#     ]
#   },
#   {
#     "id": 8,
#     "action": "Stack turquoise chair",
#     "dependencies": [
#       7
#     ]
#   },
#   {
#     "id": 9,
#     "action": "Walk to red chair 1",
#     "dependencies": [
#       4
#     ]
#   },
#   {
#     "id": 10,
#     "action": "Pick up red chair 1",
#     "dependencies": [
#       9
#     ]
#   },
#   {
#     "id": 11,
#     "action": "Transport red chair 1",
#     "dependencies": [
#       10
#     ]
#   },
#   {
#     "id": 12,
#     "action": "Stack red chair 1",
#     "dependencies": [
#       11
#     ]
#   },
#   {
#     "id": 13,
#     "action": "Walk to red chair 2",
#     "dependencies": [
#       4
#     ]
#   },
#   {
#     "id": 14,
#     "action": "Pick up red chair 2",
#     "dependencies": [
#       13
#     ]
#   },
#   {
#     "id": 15,
#     "action": "Transport red chair 2",
#     "dependencies": [
#       14
#     ]
#   },
#   {
#     "id": 16,
#     "action": "Stack red chair 2",
#     "dependencies": [
#       15
#     ]
#   },
#   {
#     "id": 17,
#     "action": "Walk to red chair 3",
#     "dependencies": [
#       4
#     ]
#   },
#   {
#     "id": 18,
#     "action": "Pick up red chair 3",
#     "dependencies": [
#       17
#     ]
#   },
#   {
#     "id": 19,
#     "action": "Transport red chair 3",
#     "dependencies": [
#       18
#     ]
#   },
#   {
#     "id": 20,
#     "action": "Stack red chair 3",
#     "dependencies": [
#       19
#     ]
#   }
# ]
# """

# # Call the function to generate the graph
# create_task_graph_visualization(task_json_input, output_filename='task_action_graph')

def get_ground_truth(frame_number: int, gt_filename: str) -> Optional[Dict]:
    """
    Retrieves the ground truth state for a specific frame number from a JSON file.
    The assumption is that the ground truth remains the same until the next entry.

    Args:
        frame_number (int): The frame number to look for.
        gt_filename (str): The path to the ground truth JSON file.

    Returns:
        Optional[Dict]: The state dictionary for the given frame_number if found, 
                        otherwise None.
    """
    try:
        # Load ground truth data
        with open(gt_filename, 'r') as f:
            gt_data = json.load(f)
        
        # Sort the data by frame_number to ensure proper ordering
        gt_data.sort(key=lambda x: x.get('frame') or x.get('frame_number') or 0)
        
        # Find the appropriate ground truth entry
        # The ground truth for a frame is the last entry with frame_number <= target frame
        selected_entry = None
        
        for entry in gt_data:
            entry_frame = entry.get('frame') or entry.get('frame_number')
            if entry_frame is None:
                continue
                
            if entry_frame <= frame_number:
                selected_entry = entry
            else:
                # Since data is sorted, we can break once we exceed the target frame
                break
        
        if selected_entry is not None:
            return selected_entry.get('state', {})
        else:
            print(f"No ground truth found for frame {frame_number}")
            return None
            
    except FileNotFoundError as e:
        print(f"Ground truth file not found: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing ground truth JSON: {e}")
        return None
    except Exception as e:
        print(f"Error getting ground truth for frame {frame_number}: {e}")
        return None

def convert_stack_gt_to_standard_format(input_json_path: str, output_json_path: str = None) -> str:
    """
    Convert Stack dataset ground truth JSON format to standard format.
    
    Moves all fields except 'frame_number' into a 'state' object and renames
    'frame_number' to 'frame' to match the standard format used in S02A08I21_gt.json.
    
    Args:
        input_json_path (str): Path to the input JSON file (Stack format)
        output_json_path (str): Path to save the converted JSON file. 
                               If None, saves as input_path with '_new' suffix
                               
    Returns:
        str: Path to the output file
    """
    import json
    import os
    
    # Read the input JSON
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    # Convert each entry
    converted_data = []
    for entry in data:
        # Extract frame_number and rename to frame
        frame_number = entry.pop('frame_number')
        
        # Create new entry with standard format
        new_entry = {
            "frame": frame_number,
            "state": entry  # All other fields go into state
        }
        converted_data.append(new_entry)
    
    # Determine output path
    if output_json_path is None:
        base_name = os.path.splitext(input_json_path)[0]
        output_json_path = f"{base_name}_new.json"
    
    # Write the converted JSON
    with open(output_json_path, 'w') as f:
        json.dump(converted_data, f, indent=2)
    
    print(f"Converted {input_json_path} to standard format and saved as {output_json_path}")
    return output_json_path

def condense_gt(input_json_path: str, output_json_path: str = None) -> str:
    """
    Condense ground truth JSON by removing entries where the state does not change.
    Keeps only entries where the state has changed from the previous entry.
    
    Args:
        input_json_path (str): Path to the input ground truth JSON file
        output_json_path (str): Path to save the condensed JSON file. 
                               If None, saves as input_path with '_condensed' suffix
                               
    Returns:
        str: Path to the output file
    """
    import json
    import os
    
    # Read the input JSON
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    if not data:
        print(f"Warning: Empty data in {input_json_path}")
        return input_json_path
    
    # Always keep the first entry
    condensed_data = [data[0]]
    previous_state = data[0].get('state', {})
    
    # Compare each subsequent entry with the previous state
    for i in range(1, len(data)):
        current_entry = data[i]
        current_state = current_entry.get('state', {})
        
        # Check if the current state is different from the previous state
        if current_state != previous_state:
            condensed_data.append(current_entry)
            previous_state = current_state
    
    # Determine output path
    if output_json_path is None:
        base_name = os.path.splitext(input_json_path)[0]
        output_json_path = f"{base_name}_condensed.json"
    
    # Write the condensed JSON
    with open(output_json_path, 'w') as f:
        json.dump(condensed_data, f, indent=2)
    
    original_count = len(data)
    condensed_count = len(condensed_data)
    reduction_percent = ((original_count - condensed_count) / original_count) * 100
    
    print(f"Condensed {input_json_path}:")
    print(f"  Original entries: {original_count}")
    print(f"  Condensed entries: {condensed_count}")
    print(f"  Reduction: {reduction_percent:.1f}%")
    print(f"  Saved as: {output_json_path}")
    
    return output_json_path

def delete_field_from_json(input_json_path: str, field_name: str, output_json_path: str = None) -> str:
    """
    Delete a specified field from a JSON file. The field will be removed if it exists
    at the top level of each entry or within the 'state' object of each entry.
    
    Args:
        input_json_path (str): Path to the input JSON file
        field_name (str): Name of the field to delete
        output_json_path (str): Path to save the modified JSON file. 
                               If None, saves as input_path with '_no_{field_name}' suffix
                               
    Returns:
        str: Path to the output file
    """
    import json
    import os
    
    # Read the input JSON
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    if not data:
        print(f"Warning: Empty data in {input_json_path}")
        return input_json_path
    
    deleted_count = 0
    
    # Process each entry in the JSON array
    for entry in data:
        # Check and delete field at top level
        if field_name in entry:
            del entry[field_name]
            deleted_count += 1
        
        # Check and delete field within 'state' object
        if 'state' in entry and isinstance(entry['state'], dict):
            if field_name in entry['state']:
                del entry['state'][field_name]
                deleted_count += 1
    
    # Determine output path
    if output_json_path is None:
        base_name = os.path.splitext(input_json_path)[0]
        extension = os.path.splitext(input_json_path)[1]
        output_json_path = f"{base_name}_no_{field_name}{extension}"
    
    # Write the modified JSON
    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Deleted field '{field_name}' from {input_json_path}:")
    print(f"  Total deletions: {deleted_count}")
    print(f"  Saved as: {output_json_path}")
    
    return output_json_path

def populate_immediate_next_step(input_json_path: str, output_json_path: str = None) -> str:
    """
    Load a JSON file and populate the 'immediate_next_step' field in each entry's 'state' object.
    The immediate_next_step is determined by looking for the next entry chronologically (by frame_number)
    where the steps_in_progress value actually changes, and taking the first step from that new list.
    If steps_in_progress changes to an empty list, continue looking until a non-empty list is found.
    
    Args:
        input_json_path (str): Path to the input JSON file
        output_json_path (str): Path to save the modified JSON file. 
                               If None, saves as input_path with '_with_next_step' suffix
                               
    Returns:
        str: Path to the output file
    """
    import json
    import os
    
    # Read the input JSON
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    if not data:
        print(f"Warning: Empty data in {input_json_path}")
        return input_json_path
    
    # Sort the data by frame_number to ensure chronological order
    data.sort(key=lambda x: x.get('frame') or x.get('frame_number') or 0)
    
    populated_count = 0
    
    # Process each entry
    for i, entry in enumerate(data):
        # Ensure the entry has a state object
        if 'state' not in entry:
            entry['state'] = {}
        
        # Get current steps_in_progress
        current_steps_in_progress = entry.get('state', {}).get('steps_in_progress', [])
        
        # Initialize immediate_next_step as None
        immediate_next_step = None
        
        # Look ahead chronologically to find when steps_in_progress changes to a non-empty value
        for j in range(i + 1, len(data)):
            next_entry = data[j]
            next_steps_in_progress = next_entry.get('state', {}).get('steps_in_progress', [])
            
            # Check if steps_in_progress has changed
            if next_steps_in_progress != current_steps_in_progress:
                # If the new steps_in_progress is not empty, take the first step and break
                if next_steps_in_progress:
                    immediate_next_step = next_steps_in_progress[0]
                    break
                # If it's empty, continue looking for the next non-empty change
                current_steps_in_progress = next_steps_in_progress
        
        # Populate the immediate_next_step field
        entry['state']['immediate_next_step'] = immediate_next_step
        
        if immediate_next_step is not None:
            populated_count += 1
    
    # Determine output path
    if output_json_path is None:
        base_name = os.path.splitext(input_json_path)[0]
        extension = os.path.splitext(input_json_path)[1]
        output_json_path = f"{base_name}_with_next_step{extension}"
    
    # Write the modified JSON
    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    total_entries = len(data)
    no_next_step_count = total_entries - populated_count
    
    print(f"Populated immediate_next_step field in {input_json_path}:")
    print(f"  Total entries: {total_entries}")
    print(f"  Entries with immediate_next_step: {populated_count}")
    print(f"  Entries with no future step changes: {no_next_step_count}")
    print(f"  Saved as: {output_json_path}")
    
    return output_json_path

