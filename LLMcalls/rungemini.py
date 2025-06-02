import os
import mimetypes
from google import genai
from google.genai import types

def save_binary_file(file_name, data):
    with open(file_name, "wb") as f:
        f.write(data)
    print(f"File saved to to: {file_name}")

def generate_history(existing, new):
    """
    Update the concise history of human actions based on the latest LLM response.

    Args:
        existing (str): The current concise history of actions performed by the human operator.
        new (str): The latest LLM response describing the current or most recent action.

    Returns:
        str: The updated concise history.
    """
    import os
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )
    model = "gemini-2.0-flash-preview-image-generation"

    # Compose the prompt for updating the history
    # Initial system prompt and task context (structured like generate)
    contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text="""Your Role: You are an AI assistant specialized in maintaining a concise, running history of what actions the human operator has performed so far in the task.
    Task: Given the existing history and the latest model response, update the history to reflect the most recent actions.
    Instructions: Be concise, avoid redundancy, and only include meaningful changes or transitions.
    Relevant Task Context:
    - The operator's overall goal is to organize the room by stacking several plastic chairs in a designated location against the wall and placing a toolbox onto a small folding table.
    - Actions include: standing idle, walking to locations, reaching for objects, picking up objects, transporting objects, placing objects, and turning towards locations or objects.
    - Transitions often follow: Walk -> Reach -> Pick up -> Transport -> Place.

    Now, update the history below.

    Existing history:
    """ + str(existing) + """

    Latest model response:
    """ + str(new) + """

    Updated concise history:"""),
                ],
            )
        ]

    generate_content_config = types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"] ,
        response_mime_type="text/plain",
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
        full_text += chunk.text if hasattr(chunk, "text") else str(chunk.candidates[0].content.parts[0].text)
    return full_text

def generate(prompt, image_path, history=None):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    # Upload the current image
    uploaded_image = client.files.upload(file=image_path)
    model = "gemini-2.0-flash-preview-image-generation"

    # Initial system prompt and initial image (only for the first call)
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""Your Role: You are an AI assistant specialized in real-time human action analysis and state tracking from video frames.
Your Task: You will be provided with a sequence of video frames. For each frame in the sequence, you must:
Analyze the frame in the context of the overall task and recent frames (if available).
Determine the current state of the task.
Identify the key objects visible in the frame.
Anticipate the human's immediate next micro-action or transition.
Conditionally, if you can confidently predict the human's more substantial action and target location over the next approximate 1-2 seconds, provide that prediction.
Use the provided Task Context Description below as your knowledge base regarding the overall goal, valid states, possible actions, key objects, and environmental layout.
Input
Video Data: A sequence of individual video frames, provided sequentially after every 0.2 seconds. You should process each frame as it arrives, potentially using information from previously processed frames in the sequence to inform your current analysis.
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
No Tools: No external tools are used; the task relies solely on the operator's manual actions. Here is the initial image"""),
                types.Part.from_uri(
                    file_uri=uploaded_image.uri,
                    mime_type=uploaded_image.mime_type,
                ),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="""```json
{
  "time": 0.0,
  "toolbox_placed_on_table": false,
  "num_chairs_stacked": 0,
  "operator_holding": "None",
  "gaze_target": "Chairs in Initial Object Area",
  "current_target_object": "Toolbox",
  "current_phase": "Toolbox Phase",
  "Identified Key Objects": "Operator standing idle in front of Initial Object Area; Toolbox (orange and black) on a turquoise chair within the Initial Object Area (mid-ground, operator's right); Multiple Red Chairs (3) and Turquoise Chairs (2 total) visible in Initial Object Area; Folding Table visible in foreground right.",
  "Expected Immediate Next Action": "Turn towards toolbox"
}
```"""),
            ],
        ),
    ]

    # Add history (text only, no images)
    if history:
        contents.extend(history)

    # Add the current user turn (with image and prompt)
    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_uri(
                    file_uri=uploaded_image.uri,
                    mime_type=uploaded_image.mime_type,
                ),
                types.Part.from_text(text=prompt),
            ],
        )
    )

    generate_content_config = types.GenerateContentConfig(
        response_modalities=["IMAGE", "TEXT"],
        response_mime_type="text/plain",
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

if __name__ == "__main__":
    results_store_file = "LLMcalls/store_results.txt"
    max_history = 3  # Number of previous prompt/response pairs to keep

    history = []
    memory = ""  # Start with empty memory

    with open(results_store_file, "w") as f_out:
        for second in range(0, 42):
            for fractionsecond in range(0, 5):
                timestamp = f"{second:01d}.{2*fractionsecond:01d}"
                image_path = f"/home/mani/Central/Cooking1/Stack/output_frames/cam2_cr-frame-{timestamp}.jpg"

                # Only keep the last max_history prompt/response pairs
                context_to_pass = history[-max_history*2:] if len(history) > max_history*2 else history[:]

                # Add memory to the prompt
                if memory:
                    prompt_with_memory = f"Summary of actions so far:\n{memory}\n\nHere is the image at time: {timestamp}"
                else:
                    prompt_with_memory = f"Here is the image at time: {timestamp}"

                response = generate(prompt_with_memory, image_path, history=context_to_pass)
                memory = generate_history(memory, response)

                # Add the new user/model turn to history (text only, no images)
                history.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=prompt_with_memory),
                        ],
                    )
                )
                history.append(
                    types.Content(
                        role="model",
                        parts=[
                            types.Part.from_text(text=response),
                        ],
                    )
                )

                print(f"{timestamp}\t{response}\n")
                f_out.write(f"{timestamp}\t{response}\n")