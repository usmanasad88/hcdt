import json
import re

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

input_path = "data/gemini-flash-image-gen-cooking-dag.json"
output_path = "data/gemini-flash-image-gen-cooking-dag.fixed.json"
log_to_json(input_path, output_path)
