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