import base64
import os
from google import genai
from google.genai import types
import sys
import os
from typing import Optional, List, Dict
import json
import time
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.imgutils import image_to_base64
from utils.textutils import extract_json_from_response
from utils.experiment_logger import ExperimentLogger

def generate(cfg: DictConfig,
    example_image_paths: List[str], 
    test_image_paths: List[str], 
    query_prompt: str,
    next_query_prompt: str, 
    logger: ExperimentLogger = None,
    frame_number: int = 0,
    responses: Optional[List[str]] = None,
    json_data: Optional[Dict[str, str]] = None):

    encoded_example_images = []
    for img_path in example_image_paths:
        encoded_img = image_to_base64(img_path, target_largest_dimension=None)
        encoded_example_images.append(encoded_img) # Will store None if encoding failed

    encoded_test_images = []
    for img_path in test_image_paths:
        encoded_img = image_to_base64(img_path, target_largest_dimension=None)
        encoded_test_images.append(encoded_img) # Will store None if encoding failed
    
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )
    
    model=cfg.model

    start_time = time.time()
    
    current_responses = responses if responses is not None else []
    contents = []

    # APPROACH 1
    # Build context from provided image-response pairs (history)
    # for i in range(len(image_paths)):
    #     prompt_for_turn = query_prompt if i == 0 else next_query_prompt
        
    #     user_parts = [types.Part.from_text(text=prompt_for_turn)]
    #     if i < len(encoded_images) and encoded_images[i]:
    #         try:
    #             image_bytes = base64.b64decode(encoded_images[i])
    #             user_parts.append(types.Part.from_bytes(
    #                 mime_type="image/jpeg", # Assuming JPEG, adjust if type varies
    #                 data=image_bytes,
    #             ))
    #         except Exception as e:
    #             print(f"Warning: Could not decode or use image {image_paths[i]} for history turn {i}: {e}")
    #     if i > 0:
    #         contents.append(types.Content(role="model", parts=[types.Part.from_text(text=current_responses[i-1])]))        
    #     contents.append(types.Content(role="user", parts=user_parts))


    # APPROACH 2
    # Build context from a complete example, and the history of frames till current frame.
    
    # first_prompt_part = [types.Part.from_text(text=query_prompt)]
    # contents.append(types.Content(role="user", parts=first_prompt_part))
    
    # # Add JSON files if provided
    # if json_data:
    #     for json_name, json_content in json_data.items():
    #         json_bytes = base64.b64encode(json_content.encode('utf-8')).decode('utf-8')
    #         contents.append(types.Content(
    #             role="user",
    #             parts=[types.Part.from_bytes(
    #                 mime_type="application/json",
    #                 data=base64.b64decode(json_bytes),
    #             )]
    #         ))

    # for i in range(len(encoded_example_images)):
    #     example_images_bytes=base64.b64decode(encoded_example_images[i])
    #     contents.append(types.Content(
    #         role="user",
    #         parts=[types.Part.from_bytes(
    #             mime_type="image/png",  # Assuming JPEG, adjust if type varies
    #             data=example_images_bytes,
    #         )]
    #     ))

    # model_reponse="Okay, I have analyzed the example video. I am ready for your questions about test video."
    # contents.append(types.Content(role="model", parts=[types.Part.from_text(text=model_reponse)]))
    # contents.append(types.Content(role="user", parts=[types.Part.from_text(text=next_query_prompt)]))
    # for i in range(len(encoded_test_images)):
    #     example_images_bytes=base64.b64decode(encoded_test_images[i])
    #     contents.append(types.Content(
    #         role="user",
    #         parts=[types.Part.from_bytes(
    #             mime_type="image/png",  # adjust if type varies
    #             data=example_images_bytes,
    #         )]
    #     ))

    # Turn 1: User provides the main prompt, JSON data, and example images
    first_user_parts = [types.Part.from_text(text=query_prompt)]
    
    # Add JSON files if provided
    if json_data:
        for json_name, json_content in json_data.items():
            json_bytes = base64.b64encode(json_content.encode('utf-8'))
            first_user_parts.append(types.Part.from_bytes(
                mime_type="application/json",
                data=json_bytes
            ))

    # Add example images
    for encoded_image in encoded_example_images:
        if encoded_image:
            image_bytes = base64.b64decode(encoded_image)
            first_user_parts.append(types.Part.from_bytes(
                mime_type="image/png",
                data=image_bytes
            ))
    
    contents.append(types.Content(role="user", parts=first_user_parts))

    # Turn 2: Model acknowledges the examples
    model_reponse="Okay, I have analyzed the example video and supporting documents. I am ready for your questions about the test video."
    contents.append(types.Content(role="model", parts=[types.Part.from_text(text=model_reponse)]))

    # Turn 3: User provides the next prompt and test images
    second_user_parts = [types.Part.from_text(text=next_query_prompt)]
    for encoded_image in encoded_test_images:
        if encoded_image:
            image_bytes = base64.b64decode(encoded_image)
            second_user_parts.append(types.Part.from_bytes(
                mime_type="image/png",
                data=image_bytes
            ))

    contents.append(types.Content(role="user", parts=second_user_parts))

    generate_content_config = types.GenerateContentConfig(
        thinking_config = types.ThinkingConfig(
            thinking_budget=1,),
        response_mime_type="text/plain",
        temperature=0.5,  
        top_p=0.95,        
        top_k=30 
        # response_mime_type="application/json",

    )
    full_response = []
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if chunk.text is not None: # Check if chunk.text is not None
            print(chunk.text, end="")
            full_response.append(chunk.text)

    response_text = "".join(full_response)
    
    # End timing and log if logger is provided
    end_time = time.time()
    generation_duration = end_time - start_time
    
    if logger:
        # Create combined prompt for token counting
        combined_prompt = f"{query_prompt}\n\n{next_query_prompt}"
        
        logger.log_generation(
            frame_number=frame_number,
            prompt=combined_prompt,
            response=response_text,
            duration=generation_duration,
            model=model,
            example_images_count=len(example_image_paths),
            test_images_count=len(test_image_paths),
            temperature=generate_content_config.temperature,
            top_p=generate_content_config.top_p,
            top_k=generate_content_config.top_k
        )
    
    return response_text

def get_ground_truth(frame_number: int, gt_filename: str) -> Optional[Dict]:
    """
    Retrieves the ground truth state for a specific frame number from a JSON file.

    Args:
        frame_number (int): The frame number to look for.
        gt_filename (str): The path to the ground truth JSON file.

    Returns:
        Optional[Dict]: The state dictionary for the given frame_number if found, 
                        otherwise None.
    """
    try:
        with open(gt_filename, 'r') as f:
            gt_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Ground truth file not found: {gt_filename}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from ground truth file: {gt_filename}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading GT file '{gt_filename}': {e}")
        return None

    if not isinstance(gt_data, list):
        print(f"Error: Ground truth data in {gt_filename} is not a list.")
        return None

    for record in gt_data:
        if isinstance(record, dict) and record.get("frame") == frame_number:
            state = record.get("state")
            if isinstance(state, dict):
                return state
            else:
                print(f"Warning: Record for frame {frame_number} in {gt_filename} has no 'state' dictionary.")
                return None # Or handle as an error depending on strictness
    
    # print(f"Warning: Frame {frame_number} not found in ground truth file: {gt_filename}")
    return None

def runPIPS(case_study, test_image_dir, total_frames_to_process='all', frame_step=15):
    use_ground_truth=True
    gt_filename="data/HAViD/S13A11I21_gt.json"

    script_dir = os.path.dirname(__file__)
    dag_file_path = os.path.join(script_dir, '..', 'data', case_study, 'dag.json')
    state_file_path = os.path.join(script_dir, '..', 'data', case_study, 'state.json')
    result_store_path= os.path.join(script_dir, '..', 'data', case_study, 'result_test.json')
    with open(dag_file_path, 'r') as f:
            task_graph_string = f.read()
    with open(state_file_path, 'r') as f:
            state_schema_string = f.read()

    prompt_text = f"""
    You are an AI assistant analyzing video frames of a person performing a task.
    Your goal is to update the state variables based on the provided task graph, state schema, and the visual information from the image.

    Task Graph Definition:
    ```json
    {task_graph_string}
    ```

    State Variables Schema:
    ```json
    {state_schema_string}
    ```

    Instruction:
    Based on the image provided and the schemas above, update the state variables.
    Boolean state variables are not strictly boolean and can be False, True or Unknown.
    Output the updated state variables as a JSON object.
    Ensure your output is only the JSON object representing the updated state variables.
    """    
    if total_frames_to_process == 'all':
        total_frames_to_process = len([f for f in os.listdir(test_image_dir) if f.endswith('.png')])
    all_responses_data = []
    current_response = None
    frame_num = 301
    while frame_num < total_frames_to_process:
        current_frame_path = os.path.join(test_image_dir, f"frame_{frame_num:04d}.png")

        if frame_num == 1:
            current_response = generate([current_frame_path], prompt_text)
            response_obj = json.loads(extract_json_from_response(current_response))     
            all_responses_data.append({"frame": frame_num, "state": response_obj})        

        else:
            prev_frame = frame_num - frame_step
            prev_frame_path = os.path.join(test_image_dir, f"frame_{prev_frame:04d}.png")

            call_image_paths = [prev_frame_path, current_frame_path]
            call_responses = [current_response]
            if use_ground_truth:
                call_responses = [json.dumps(get_ground_truth(prev_frame, gt_filename))]
            current_response = generate(call_image_paths, prompt_text, responses=call_responses)              
            response_obj = json.loads(extract_json_from_response(current_response))
            all_responses_data.append({"frame": frame_num, "state": response_obj})        
        
        current_response = f'"""{json.dumps(response_obj)}"""'

        all_responses_data.sort(key=lambda x: x.get("frame", float('inf'))) # Sort by frame number
        with open(result_store_path, 'w') as f:
            json.dump(all_responses_data, f, indent=4)
        frame_num += frame_step

def runICL_HI(cfg: DictConfig):
    
    # Initialize experiment logger
    output_dir = os.getcwd()
    output_dir = os.path.join(output_dir, 'logs')
    logger = ExperimentLogger(output_dir=output_dir)
    
    # Start experiment with notes
    experiment_notes = f"ICL experiment on {cfg.case_study} dataset"
    experiment_id = logger.start_experiment(cfg, experiment_notes)
    
    script_dir = os.path.dirname(__file__)
    dag_file_path = os.path.join(script_dir, '..', 'data', cfg.case_study, 'dag.json')
    state_file_path = os.path.join(script_dir, '..', 'data', cfg.case_study, 'state.json')
    # example_gt_path=  os.path.join(script_dir, '..', 'data', case_study, 'S02A08I21_gt.json')
    result_store_path = os.path.join(script_dir, '..', 'data', cfg.case_study, cfg.exp.type + '_result.json')
    
    with open(dag_file_path, 'r') as f:
            task_graph_string = f.read()
    with open(state_file_path, 'r') as f:
            state_schema_string = f.read()
    with open(cfg.exp.example_gt_filename, 'r') as f:
            example_gt_string = f.read()

    json_data = {
        "task_graph": task_graph_string,
        "state_schema": state_schema_string,
        "example_gt": example_gt_string
    }

    prompt_template = cfg.prompts.version2
    if cfg.exp.prompt_version == "version1":
        prompt_text = prompt_template.format(
            task_graph_string=task_graph_string,
            state_schema_string=state_schema_string,
            example_gt_string=example_gt_string,
            example_frame_step=cfg.example_frame_step,
            fps=cfg.fps,
            test_frame_step=cfg.test_frame_step,
            example_frame_step_seconds=cfg.example_frame_step / cfg.fps,
            test_frame_step_seconds=cfg.test_frame_step / cfg.fps
        )
    elif cfg.exp.prompt_version == "version2":
        prompt_text = prompt_template.format(
            example_frame_step_seconds=cfg.example_frame_step / cfg.fps,
            test_frame_step_seconds=cfg.test_frame_step / cfg.fps
        )
    example_image_paths = []
    example_frame_files = sorted([f for f in os.listdir(cfg.exp.example_image_dir) if f.endswith('.png')])
    for i in range(0, len(example_frame_files), cfg.example_frame_step):
        example_image_paths.append(os.path.join(cfg.exp.example_image_dir, example_frame_files[i]))

    if cfg.total_frames_to_process == 'all':
        total_frames_to_process = len([f for f in os.listdir(cfg.exp.test_image_dir) if f.endswith('.png')])
    all_responses_data = []
    
    current_response = None
    frame_num = cfg.start_frame

    

    try:
        while frame_num < total_frames_to_process:
            if cfg.end_frame > 0 and frame_num >= cfg.end_frame:
                break

            second_prompt_text = f"History of frames uptil {cfg.test_frame_step / cfg.fps:.2f} second(s) at frame number {frame_num} at {cfg.fps} fps :"
            test_image_paths = []
            test_frame_files = sorted([f for f in os.listdir(cfg.exp.test_image_dir) if f.endswith('.png')])
            for i in range(0, frame_num, cfg.test_frame_step):
                test_image_paths.append(os.path.join(cfg.exp.test_image_dir, test_frame_files[i]))

            # if frame_num == 1:
            current_response = generate(cfg, example_image_paths, test_image_paths, prompt_text, second_prompt_text, logger, frame_num, json_data=json_data)
            response_obj = json.loads(extract_json_from_response(current_response))     
            all_responses_data.append({"frame": frame_num, "state": response_obj})        

            # else:
            #     prev_frame = frame_num - cfg.test_frame_step
            #     prev_frame_path = os.path.join(cfg.test_image_dir, f"frame_{prev_frame:04d}.png")

            #     call_image_paths = [prev_frame_path, current_frame_path]
            #     call_responses = [current_response]
            #     if cfg.use_ground_truth:
            #         call_responses = [json.dumps(get_ground_truth(prev_frame, cfg.gt_filename))]
            #     current_response = generate(call_image_paths, prompt_text, second_prompt_text, responses=call_responses)              
            #     response_obj = json.loads(extract_json_from_response(current_response))
            #     all_responses_data.append({"frame": frame_num, "state": response_obj})        
            
            current_response = f'"""{json.dumps(response_obj)}"""'

            all_responses_data.sort(key=lambda x: x.get("frame", float('inf'))) # Sort by frame number
            with open(result_store_path, 'w') as f:
                json.dump(all_responses_data, f, indent=4)
            frame_num += cfg.test_frame_step
            
            # Save checkpoint every 5 generations
            if len(all_responses_data) % 5 == 0:
                logger.save_checkpoint()
                
        # Save the config for reproducibility
        with open(os.path.join(output_dir, "config_used.yaml"), 'w') as f:
            f.write(OmegaConf.to_yaml(cfg))
            
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        raise
    finally:
        # Always end the experiment to save logs
        logger.end_experiment()

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))  # <-- Add this line


    # runPIPS(case_study, test_image_dir)
    if cfg.exp.type=="ICL":
        runICL_HI(cfg)

if __name__ == "__main__":

    # use_ground_truth=False
    # gt_filename="data/HAViD/S13A11I21_gt.json"
    # test_image_dir = "/home/mani/Central/HaVid/ExampleContext/all_frames_f" # S13 A11 I21
    # example_image_dir = "home/mani/Central/HaVid/HAViD/S02A08I21S1/frames" # S02 A08 I21 S1 S02A08I21_gt
    # case_study="HAViD"
    # runPIPS(case_study,test_image_dir)
    # runICL_HI(case_study,test_image_dir,example_image_dir,gt_filename,use_ground_truth=use_ground_truth)
    main()
    