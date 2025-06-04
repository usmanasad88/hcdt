import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from LLMcalls.runllama import run_ollama_llama_vision
from LLMcalls.rungemma import run_llama_mtmd


def create_experiment_snapshot(experiment_name, frame_cutoff):
    """
    Processes annotation files for a given experiment and frame number,
    and saves the information up to that frame in a JSON file.

    Args:
        experiment_name (str): The name of the experiment (e.g., "S01A04I01").
        frame_cutoff (int): The frame number up to which data should be included.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_filename = os.path.join(script_dir, f"{experiment_name}_{frame_cutoff}.json")

    output_data = {
        "experiment_name": experiment_name,
        "frame_cutoff": frame_cutoff,
        "annotations": {
            "lh": {
                "aa": {"collaboration": [], "temporal": []},
                "pt": {"collaboration": [], "temporal": []}
            },
            "rh": {
                "aa": {"collaboration": [], "temporal": []},
                "pt": {"collaboration": [], "temporal": []}
            }
        }
    }

    hands = ["lh", "rh"]
    action_types_short = ["aa", "pt"] # Atomic Action, Primitive Task

    for hand in hands:
        for action_type in action_types_short:
            view_hand_action_dir = f"view0_{hand}_{action_type}"
            base_filename_prefix = f"{experiment_name}_{view_hand_action_dir}"

            # Process collaboration timestamps
            collab_filename = f"{base_filename_prefix}_collaboration.txt"
            collab_filepath = os.path.join(script_dir, view_hand_action_dir, "collaboration_timestamps", collab_filename)
            
            current_collab_list = []
            if os.path.exists(collab_filepath):
                with open(collab_filepath, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5: # Expected format: start end type action1 action2
                            try:
                                start_frame, end_frame = int(parts[0]), int(parts[1])
                                if end_frame <= frame_cutoff:
                                    entry = {
                                        "start_frame": start_frame,
                                        "end_frame": end_frame,
                                        "type": parts[2],
                                        "left_action_label": parts[3],
                                        "right_action_label": parts[4]
                                    }
                                    current_collab_list.append(entry)
                            except ValueError:
                                print(f"Warning: Skipping malformed line in {collab_filepath}: {line.strip()}")
                        elif len(parts) > 0 : # Avoid error on empty lines, but log if not empty and not 5 parts
                            print(f"Warning: Skipping line with unexpected format in {collab_filepath}: {line.strip()}")
            else:
                print(f"Warning: File not found {collab_filepath}")
            output_data["annotations"][hand][action_type]["collaboration"] = current_collab_list

            # Process temporal timestamps
            temporal_filename = f"{base_filename_prefix}_temporal.txt"
            temporal_filepath = os.path.join(script_dir, view_hand_action_dir, "temporal_timestamps", temporal_filename)
            
            current_temporal_list = []
            if os.path.exists(temporal_filepath):
                with open(temporal_filepath, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 3: # Expected format: start end action
                            try:
                                start_frame, end_frame = int(parts[0]), int(parts[1])
                                if end_frame <= frame_cutoff:
                                    entry = {
                                        "start_frame": start_frame,
                                        "end_frame": end_frame,
                                        "action_label": parts[2]
                                    }
                                    current_temporal_list.append(entry)
                            except ValueError:
                                print(f"Warning: Skipping malformed line in {temporal_filepath}: {line.strip()}")
                        elif len(parts) > 0:
                             print(f"Warning: Skipping line with unexpected format in {temporal_filepath}: {line.strip()}")
            else:
                print(f"Warning: File not found {temporal_filepath}")
            output_data["annotations"][hand][action_type]["temporal"] = current_temporal_list

    with open(output_filename, 'w') as outfile:
        json.dump(output_data, outfile, indent=4)
    
    print(f"Successfully created snapshot: {output_filename}")

def rearrange_json_by_frame(input_filename, output_filename):
    """
    Rearranges the JSON data by frame number, grouping annotations by frame.

    Args:
        input_filename (str): The input JSON file with annotations.
        output_filename (str): The output JSON file to save rearranged data.
    """
    with open(input_filename, 'r') as infile:
        data = json.load(infile)

    rearranged_data = {}
    
    for hand in data["annotations"]:
        for action_type in data["annotations"][hand]:
            for annotation_type in data["annotations"][hand][action_type]:
                for entry in data["annotations"][hand][action_type][annotation_type]:
                    start_frame = entry.get("start_frame")
                    end_frame = entry.get("end_frame")
                    
                    if start_frame is not None and end_frame is not None:
                        for frame in range(start_frame, end_frame + 1):
                            if frame not in rearranged_data:
                                rearranged_data[frame] = []
                            rearranged_data[frame].append({
                                "hand": hand,
                                "action_type": action_type,
                                "annotation_type": annotation_type,
                                "entry": entry
                            })

    with open(output_filename, 'w') as outfile:
        json.dump(rearranged_data, outfile, indent=4)
    
    print(f"Successfully rearranged data by frame: {output_filename}")

def generate_action_summary(havid_data_str, dictionary_data_str):
    """
    Generates a natural language summary of actions from HAVID data
    without using an LLM.
    """
    # havid_data = json.loads(havid_data_str)
    havid_data = havid_data_str
    # dictionary_data = json.loads(dictionary_data_str)
    dictionary_data = dictionary_data_str

    frame_cutoff = havid_data.get("frame_cutoff", float('inf'))

    # 1. Create lookup maps from the dictionary
    verb_map = {item['code']: item['label'] for item in dictionary_data['action_verbs']}
    object_map = {item['code']: item['label'] for item in dictionary_data['objects']}
    # Add tools to object_map as they can be objects of actions
    for item in dictionary_data.get('tools', []):
        object_map[item['code']] = item['label']


    def parse_action_label(action_label_code):
        """Parses an action code like 'pckbx' into verb and objects."""
        if action_label_code == "null" or not action_label_code:
            return "is paused", []

        verb_code = action_label_code[0]
        obj_codes = []
        if len(action_label_code) > 1:
            # Objects are typically 2-character codes
            for i in range(1, len(action_label_code), 2):
                obj_codes.append(action_label_code[i:i+2])

        verb_str = verb_map.get(verb_code, verb_code) # Fallback to code if not found
        obj_strs = [object_map.get(oc, oc) for oc in obj_codes] # Fallback to code

        return verb_str, obj_strs

    # 2. Collect and parse actions for each hand
    all_actions = []
    for hand_code in ['lh', 'rh']:
        hand_name = "left" if hand_code == 'lh' else "right"
        if hand_code in havid_data['annotations']:
            # Using atomic actions ('aa') as per the example
            temporal_actions = havid_data['annotations'][hand_code].get('aa', {}).get('temporal', [])
            for action_entry in temporal_actions:
                if action_entry['start_frame'] < frame_cutoff:
                    verb, objects = parse_action_label(action_entry['action_label'])
                    all_actions.append({
                        'hand': hand_name,
                        'start': action_entry['start_frame'],
                        'end': action_entry['end_frame'],
                        'verb': verb,
                        'objects': objects,
                        'original_label': action_entry['action_label'] # for debugging or complex rules
                    })

    # Sort all actions chronologically
    all_actions.sort(key=lambda x: x['start'])

    # 3. Group actions and build summary sentences
    summary_sentences = []
    
    # Handle initial pause
    initial_lh_paused = True
    initial_rh_paused = True
    for action in all_actions:
        if action['start'] < 27: # Based on example, pause ends around frame 26
             if action['hand'] == 'left' and action['verb'] != 'is paused':
                 initial_lh_paused = False
             if action['hand'] == 'right' and action['verb'] != 'is paused':
                 initial_rh_paused = False
        else: # No need to check further if past the initial phase
            if action['hand'] == 'left' and action['verb'] != 'is paused': initial_lh_paused = False
            if action['hand'] == 'right' and action['verb'] != 'is paused': initial_rh_paused = False
            break # Optimization: if any hand is active early, no need to scan all initial nulls

    # Check if the very first actions are null
    first_lh_action = next((a for a in all_actions if a['hand'] == 'left'), None)
    first_rh_action = next((a for a in all_actions if a['hand'] == 'right'), None)

    if (first_lh_action and first_lh_action['verb'] == 'is paused' and first_lh_action['end'] > 0) and \
       (first_rh_action and first_rh_action['verb'] == 'is paused' and first_rh_action['end'] > 0):
        summary_sentences.append("Initially, both hands are paused.")


    # Process actual actions
    processed_actions_indices = set()
    
    # This logic aims to create 2-3 sentences, focusing on significant action blocks.
    # It's a heuristic approach.

    current_sentence_hand = None
    current_sentence_parts = [] # Stores verb-object phrases for the current hand's segment

    # Iterate through sorted actions to build segments
    segments = []
    if not all_actions:
        if not summary_sentences: # If no initial pause and no actions
             return "No actions recorded within the timeframe."
        else: # Only initial pause
             return " ".join(summary_sentences)


    active_segment = []
    last_hand_for_segment = all_actions[0]['hand'] if all_actions[0]['verb'] != 'is paused' else None

    for i, action in enumerate(all_actions):
        if action['verb'] == 'is paused':
            if active_segment: # End current segment if a pause interrupts
                segments.append({'hand': last_hand_for_segment, 'actions': list(active_segment)})
                active_segment = []
            last_hand_for_segment = None # Reset hand during pause
            continue

        if last_hand_for_segment and action['hand'] != last_hand_for_segment:
            if active_segment:
                segments.append({'hand': last_hand_for_segment, 'actions': list(active_segment)})
                active_segment = []
        
        active_segment.append(action)
        last_hand_for_segment = action['hand']
    
    if active_segment: # Add any remaining segment
        segments.append({'hand': last_hand_for_segment, 'actions': list(active_segment)})

    # Now, format these segments into sentences
    # This is where the "natural language" part gets tricky without an LLM.
    # We'll try to follow the example's structure.

    for seg_idx, segment_data in enumerate(segments):
        hand_name = segment_data['hand']
        actions = segment_data['actions']
        
        if not actions: continue

        # Condense actions in the segment
        # Example: "approaches, grasps, moves the X, and places it into Y"
        condensed_phrases = []
        k = 0
        while k < len(actions):
            current_k_action = actions[k]
            main_object = current_k_action['objects'][0] if current_k_action['objects'] else "something"
            
            verbs_on_main_object = [current_k_action['verb']]
            
            # See how many subsequent actions by the same hand operate on the same primary object
            # before a "place" or "insert" or change of object
            m = k + 1
            while m < len(actions):
                next_action = actions[m]
                if next_action['objects'] and next_action['objects'][0] == main_object and \
                   next_action['verb'] not in ['place', 'insert']: # Verbs that often change context
                    verbs_on_main_object.append(next_action['verb'])
                    m += 1
                else:
                    break # Object changed or it's a placing action

            # Format the collected verbs for the main_object
            if len(verbs_on_main_object) > 1:
                verb_combo_str = ", ".join(v for v in verbs_on_main_object[:-1]) + "s, and " + verbs_on_main_object[-1] + "s"
            else:
                verb_combo_str = verbs_on_main_object[0] + "s"
            
            phrase = f"{verb_combo_str} the {main_object}"

            # Check if the last action in this sub-group was a place/insert
            last_action_in_sub_group = actions[m-1]
            if last_action_in_sub_group['verb'] == 'place' and len(last_action_in_sub_group['objects']) == 2:
                # The verb_combo_str already includes "places". We need to adjust.
                # Rebuild verb_combo_str without the final "place"
                if len(verbs_on_main_object) > 1 and verbs_on_main_object[-1] == 'place':
                     temp_verbs = verbs_on_main_object[:-1]
                     if temp_verbs:
                         verb_combo_str = (", ".join(v for v in temp_verbs[:-1]) + "s, and " if len(temp_verbs) > 1 else "") + temp_verbs[-1] + "s"
                         phrase = f"{verb_combo_str} the {main_object}, and places it into the {last_action_in_sub_group['objects'][1]}"
                     else: # Only "place" was there
                         phrase = f"places the {main_object} into the {last_action_in_sub_group['objects'][1]}"

                elif verbs_on_main_object == ['place']: # Only action was "place"
                     phrase = f"places the {main_object} into the {last_action_in_sub_group['objects'][1]}"
                # If "place" was not the last verb in verbs_on_main_object, but the action itself is place
                # This case might need more refinement if the logic above isn't perfect.
                # For now, we assume 'place' would be the last verb if it's part of the sequence on main_object.

            elif last_action_in_sub_group['verb'] == 'insert' and len(last_action_in_sub_group['objects']) == 2:
                # Similar logic for "insert"
                if len(verbs_on_main_object) > 1 and verbs_on_main_object[-1] == 'insert':
                     temp_verbs = verbs_on_main_object[:-1]
                     if temp_verbs:
                        verb_combo_str = (", ".join(v for v in temp_verbs[:-1]) + "s, and " if len(temp_verbs) > 1 else "") + temp_verbs[-1] + "s"
                        phrase = f"{verb_combo_str} the {main_object}, and inserts it into the {last_action_in_sub_group['objects'][1]}"
                     else:
                        phrase = f"inserts the {main_object} into the {last_action_in_sub_group['objects'][1]}"
                elif verbs_on_main_object == ['insert']:
                     phrase = f"inserts the {main_object} into the {last_action_in_sub_group['objects'][1]}"


            condensed_phrases.append(phrase)
            k = m # Move main iterator past the processed sub-group

        # Join condensed phrases for the current segment
        if condensed_phrases:
            segment_text = ""
            if seg_idx == 0 and summary_sentences and "Initially" in summary_sentences[0]: # First active segment
                segment_text = f"Then, the {hand_name} hand " + ", then ".join(condensed_phrases)
            elif seg_idx > 0 and segments[seg_idx-1]['hand'] == hand_name : # Continued action by same hand
                 segment_text = f"and then " + ", then ".join(condensed_phrases) # Assumes it's part of previous sentence
            elif seg_idx > 0 : # Switch of hand or new block
                 # Check for concurrency (very simplified)
                 prev_segment_end = segments[seg_idx-1]['actions'][-1]['end'] if segments[seg_idx-1]['actions'] else 0
                 current_segment_start = actions[0]['start']
                 if current_segment_start < prev_segment_end + 10: # Arbitrary small overlap/quick succession
                     # Find the last sentence and append "while..."
                     if summary_sentences:
                         summary_sentences[-1] = summary_sentences[-1].strip('.') + f", while the {hand_name} hand " + ", then ".join(condensed_phrases)
                         segment_text = None # Handled by appending
                     else: # Should not happen if seg_idx > 0
                         segment_text = f"Meanwhile, the {hand_name} hand " + ", then ".join(condensed_phrases)

                 else:
                    segment_text = f"Subsequently, the {hand_name} hand " + ", then ".join(condensed_phrases)
            else: # First segment, no initial pause sentence
                segment_text = f"The {hand_name} hand " + ", then ".join(condensed_phrases)


            if segment_text and not segment_text.endswith('.'):
                 segment_text += "."
            if segment_text:
                 summary_sentences.append(segment_text)


    final_summary = " ".join(s for s in summary_sentences if s)
    # Basic cleanup for multiple periods or spacing issues from concatenations
    final_summary = final_summary.replace("..", ".").replace(". .", ".")
    return final_summary if final_summary else "No actions to summarize."


# --- Example Usage
exp_name= "S02A08I21"
create_experiment_snapshot(exp_name, 540)
input_filename = "data/HAViD_temporalAnnotation/havid_dictionary.json"
with open(input_filename, 'r') as infile:
    dictionary_json_input = json.load(infile)

input_filename = f"data/HAViD_temporalAnnotation/{exp_name}_540.json"
with open(input_filename, 'r') as infile:
    havid_json_input = json.load(infile)

summary = generate_action_summary(havid_json_input, dictionary_json_input)
print(summary)

# if __name__ == '__main__':
#     # Example usage:
#     # Ensure this script is in the /home/mani/Repos/hcdt/data/HAViD_temporalAnnotation/ directory
#     # and the data files are in subdirectories like view0_lh_aa/collaboration_timestamps/ etc.
    
#     # create_experiment_snapshot("S01A04I01", 100)
#     # create_experiment_snapshot("S01A04I01", 1000)
#     rearrange_json_by_frame("data/HAViD_temporalAnnotation/S01A04I01_100.json", "data/HAViD_temporalAnnotation/S01A04I01_100_rearranged.json")
#     pass
def create_prompt_for_llm(experiment_name, frame_cutoff, task_description_file, dictionary_file, snapshot_file):
    """
    Generates a prompt string based on a template.

    Args:
        experiment_name (str): The name of the experiment (e.g., "S02A08I21").
        frame_cutoff (int): The frame number up to which data was included in the snapshot.
        task_description_file (str): Path to the task description file.
        dictionary_file (str): Path to the dictionary JSON file.
        snapshot_file (str): Path to the experiment snapshot JSON file.

    Returns:
        str: The formatted prompt string.
    """

    with open(task_description_file, 'r') as f:
        task_desc = f.read()
    with open(dictionary_file, 'r') as f:
        dictionary = f.read()
    with open(snapshot_file, 'r') as f:
        snapshot = f.read()

    prompt = f"""A person is performing the following tasks:
{task_desc}
A video for the person performing these tasks is annotated as per this dictionary:
{dictionary}
Up till this point (frame {frame_cutoff}), the person has done all this:
{snapshot}
Summarize the person's actions in natural language.
"""
    return prompt
    prompt = f"""A person is performing the following tasks:
{task_description_file}
A video for the person performing these tasks is annotated as per this dictionary:
{dictionary_file}
Up till this point (frame {frame_cutoff}), the person has done all this:
{snapshot_file}
Summarize the person's actions in natural language.
"""
    return prompt

# --- Example Usage for the new prompt function ---
# Assuming exp_name is already defined as "S02A08I21" and frame_cutoff as 540
task_file = f"data/HAViD_temporalAnnotation/I21.txt" # e.g. I21.txt from S02A08I21
dictionary_filepath = "data/HAViD_temporalAnnotation/havid_dictionary.json"
snapshot_filepath = f"data/HAViD_temporalAnnotation/{exp_name}_540.json"

llm_prompt = create_prompt_for_llm(exp_name, 540, task_file, dictionary_filepath, snapshot_filepath)
# print("\n--- LLM Prompt ---")
# print(llm_prompt)

try:
    # print("\n--- LLM Summary (Ollama Llama Vision) ---")
    #llm_summary = run_ollama_llama_vision(llm_prompt)
    llm_summary = run_llama_mtmd(llm_prompt)

    print(llm_summary)
except ImportError:
    print("Could not import run_ollama_llama_vision from runollama.py. Skipping LLM summary.")
except Exception as e:
    print(f"An error occurred while running Ollama Llama Vision: {e}")

if __name__ == '__main__':
    # Example usage:
    # Ensure this script is in the /home/mani/Repos/hcdt/data/HAViD_temporalAnnotation/ directory
    # and the data files are in subdirectories like view0_lh_aa/collaboration_timestamps/ etc.
    
    # create_experiment_snapshot("S01A04I01", 100)
    # create_experiment_snapshot("S01A04I01", 1000)
    # rearrange_json_by_frame("S01A04I01_100.json", "S01A04I01_100_rearranged.json") # Adjusted paths for example
    pass
