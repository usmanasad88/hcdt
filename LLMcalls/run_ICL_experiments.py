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
from openai import OpenAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.imgutils import image_to_base64
from utils.textutils import extract_json_from_response
from utils.experiment_logger import ExperimentLogger

# Add this import at the top with your other imports
from openai import OpenAI

def generate_openAI(cfg: DictConfig,
    query_prompt: str,
    second_image_paths: Optional[List[str]] = None,   
    first_image_paths: Optional[List[str]] = None,  
    next_query_prompt: Optional[str] = None, 
    logger: ExperimentLogger = None,
    frame_number: int = 0,
    responses: Optional[List[str]] = None,
    json_data: Optional[List[str]] = None):

    # Encode test images
    encoded_test_images = []
    if second_image_paths:
        for img_path in second_image_paths:
            encoded_img = image_to_base64(img_path, target_largest_dimension=None)
            encoded_test_images.append(encoded_img)

    # Initialize OpenAI client for Gemini 
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable must be set")
    
    client = OpenAI(
        api_key=gemini_api_key,  
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    
    model = cfg.model
    start_time = time.time()
    
    # Build messages array
    messages = []
    
    # First message: User provides main prompt, JSON data, and example images
    first_message_content = [{"type": "text", "text": query_prompt}]
    
    # Add JSON data as text (OpenAI-style doesn't support file attachments the same way)
    if json_data:
        for json_content in json_data:
            first_message_content.append({
                "type": "text", 
                "text": f"\n\nJSON Data:\n{json_content}"
            })
    
    # Add example images
    if first_image_paths:
        for img_path in first_image_paths:
            encoded_img = image_to_base64(img_path, target_largest_dimension=None)
            if encoded_img:
                first_message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_img}"
                    }
                })
    
    messages.append({
        "role": "user",
        "content": first_message_content
    })
    
    # Add model acknowledgment for ICL
    if cfg.exp.type == "ICL":
        messages.append({
            "role": "assistant",
            "content": "Okay, I have analyzed the example video and supporting documents. I am ready for your questions about the test video."
        })
        
        # Add second user message with test images
        second_message_content = [{"type": "text", "text": next_query_prompt}]
        
        # Add test images
        for encoded_image in encoded_test_images:
            if encoded_image:
                second_message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_image}"
                    }
                })
        
        messages.append({
            "role": "user",
            "content": second_message_content
        })
    
    # Make the API call
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.5,
            top_p=0.95,
            max_tokens=4096,
            stream=True
        )
        
        full_response = []
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="")
                full_response.append(content)
        
        response_text = "".join(full_response)
        
    except Exception as e:
        print(f"Error in OpenAI API call: {e}")
        response_text = ""
    
    # End timing and log if logger is provided
    end_time = time.time()
    generation_duration = end_time - start_time
    
    if logger:
        # Create combined prompt for token counting
        combined_prompt = f"{query_prompt}\n\n{next_query_prompt if next_query_prompt else ''}"
        
        logger.log_generation(
            frame_number=frame_number,
            prompt=combined_prompt,
            response=response_text,
            duration=generation_duration,
            model=model,
            example_images_count=len(first_image_paths) if first_image_paths else 0,
            test_images_count=len(second_image_paths) if second_image_paths else 0,
            temperature=0.5,
            top_p=0.95,
            top_k=None  # OpenAI doesn't have top_k
        )
    
    return response_text

def generate(cfg: DictConfig,
    query_prompt: str,
    second_image_paths: Optional[List[str]] = None,   
    first_image_paths: Optional[List[str]] = None,  
    next_query_prompt: Optional[str] = None, 
    logger: ExperimentLogger = None,
    frame_number: int = 0,
    responses: Optional[List[str]] = None,
    json_data: Optional[List[str]] = None):



    encoded_test_images = []
    for img_path in second_image_paths:
        encoded_img = image_to_base64(img_path, target_largest_dimension=None)
        encoded_test_images.append(encoded_img) # Will store None if encoding failed
    
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )
    
    model=cfg.model

    start_time = time.time()
    
    current_responses = responses if responses is not None else []
    contents = []  

    # Turn 1: User provides the main prompt, JSON data, and example images
    first_user_parts = [types.Part.from_text(text=query_prompt)]
    
    # Add JSON files if provided
    if json_data:
        for json_content in json_data:
            # Direct encoding to bytes - no double encoding
            first_user_parts.append(types.Part.from_bytes(
                mime_type="application/json",
                data=json_content.encode('utf-8')  # Direct encoding, no base64
            ))

    # Add example images
    if cfg.use_file_upload:
        # Upload files first, then add to parts
        for img_path in first_image_paths:
            uploaded_file = client.files.upload(file=img_path)
            first_user_parts.append(types.Part.from_uri(file_uri=uploaded_file.uri, mime_type="image/png"))  

    else:
        encoded_example_images = []
        for img_path in first_image_paths:
            encoded_img = image_to_base64(img_path, target_largest_dimension=None)
            encoded_example_images.append(encoded_img) # Will store None if encoding failed

        if encoded_example_images:
            for encoded_image in encoded_example_images:
                if encoded_image:
                    image_bytes = base64.b64decode(encoded_image)
                    first_user_parts.append(types.Part.from_bytes(
                        mime_type="image/png",
                        data=image_bytes
                    ))
        
    contents.append(types.Content(role="user", parts=first_user_parts))

    # Turn 2: Model acknowledges the examples
    if cfg.exp.type == "ICL":
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
            example_images_count=len(first_image_paths),
            test_images_count=len(second_image_paths),
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


    if cfg.exp.prompt_version == "version1":
        prompt_template = cfg.prompts.version1

        prompt_text = prompt_template.format(
            task_graph_string=task_graph_string,
            state_schema_string=state_schema_string,
            example_gt_string=example_gt_string,
            example_frame_step=cfg.exp.example_frame_step,
            fps=cfg.exp.fps,
            test_frame_step=cfg.exp.test_frame_step,
            example_frame_step_seconds=cfg.exp.example_frame_step / cfg.exp.fps,
            test_frame_step_seconds=cfg.exp.test_frame_step / cfg.exp.fps
        )
        json_data = None  # No JSON data needed for version1 prompt

    elif cfg.exp.prompt_version == "version2":
        prompt_template = cfg.prompts.version2

        prompt_text = prompt_template.format(
            example_frame_step_seconds=cfg.exp.example_frame_step / cfg.exp.fps,
            test_frame_step_seconds=cfg.exp.test_frame_step / cfg.exp.fps
        )
        json_data = {
            "task_graph": task_graph_string,
            "state_schema": state_schema_string,
            "example_gt": example_gt_string
        }
    elif cfg.exp.prompt_version == "version3":
        prompt_template = cfg.prompts.version3

        prompt_text = prompt_template.format(
            task_graph_string=task_graph_string,
            state_schema_string=state_schema_string,
            example_frame_step=cfg.exp.example_frame_step,
            fps=cfg.exp.fps,
            test_frame_step=cfg.exp.test_frame_step,
            example_frame_step_seconds=cfg.exp.example_frame_step / cfg.exp.fps,
            test_frame_step_seconds=cfg.exp.test_frame_step / cfg.exp.fps
        )
        json_data = None  # No JSON data needed for version1 prompt
    
    first_image_paths = []

    if cfg.exp.attach_drawing:
        first_image_paths = [cfg.exp.drawing]        
        prompt_text += f"\n\n{cfg.exp.drawing_prompt}\n\n"
        
    
    example_frame_files = sorted([f for f in os.listdir(cfg.exp.example_image_dir) if f.endswith('.png')])
    for i in range(0, len(example_frame_files), cfg.exp.example_frame_step):
        first_image_paths.append(os.path.join(cfg.exp.example_image_dir, example_frame_files[i]))

    if cfg.total_frames_to_process == 'all':
        total_frames_to_process = len([f for f in os.listdir(cfg.exp.test_image_dir) if f.endswith('.png')])
    all_responses_data = []
    
    current_response = None
    frame_num = cfg.start_frame

    

    try:
        while frame_num < total_frames_to_process:
            if cfg.end_frame > 0 and frame_num >= cfg.end_frame:
                break
            second_prompt_template = cfg.second_prompt.v1
            second_prompt_text = second_prompt_template.format(                
                fps=cfg.exp.fps,
                frame_num=frame_num,
                last_frame_time=f"{frame_num / cfg.exp.fps:.2f}"
            )
            second_image_paths = []
            test_frame_files = sorted([f for f in os.listdir(cfg.exp.test_image_dir) if f.endswith('.png')])
            for i in range(0, frame_num, cfg.exp.test_frame_step):
                second_image_paths.append(os.path.join(cfg.exp.test_image_dir, test_frame_files[i]))

            if cfg.use_openai:
                current_response = generate_openAI(cfg, prompt_text, second_image_paths, first_image_paths, second_prompt_text, logger, frame_num, json_data=json_data)
            else:
                current_response = generate(cfg, prompt_text, second_image_paths, first_image_paths, second_prompt_text, logger, frame_num, json_data=json_data)
            response_obj = json.loads(extract_json_from_response(current_response))     
            all_responses_data.append({"frame": frame_num, "state": response_obj})        

              
            current_response = f'"""{json.dumps(response_obj)}"""'

            all_responses_data.sort(key=lambda x: x.get("frame", float('inf'))) # Sort by frame number
            with open(result_store_path, 'w') as f:
                json.dump(all_responses_data, f, indent=4)
            frame_num += cfg.exp.test_frame_step
            
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
    # print(OmegaConf.to_yaml(cfg))  # Print the entire config for debugging
    if cfg.exp.type=="ICL":
        runICL_HI(cfg)
    # try:
    #     json_file_path = "/home/mani/Repos/hcdt/data/stack_dag.json"
    #     with open(json_file_path, 'r') as f:
    #         json_content_string = f.read()

    #     # 3. Prepare the json_data dictionary to be passed
    #     test_json_data = [json_content_string]

    #     test_prompt = "What is the action for the object with id 8 in the attached JSON file?"
    #     response = generate(
    #             cfg,
    #             query_prompt=test_prompt,
    #             first_image_paths=[],
    #             second_image_paths=[],
    #             next_query_prompt="",
    #             json_data=test_json_data
    #         )

    #     # 6. Print the result
    #     print("\n✅ Test finished successfully!")
    #     print("LLM Response:")
    #     print("----------------")
    #     print(response)
    #     print("----------------")

    # except Exception as e:
    #     print(f"\n❌ Test failed with an error: {e}")
    #     import traceback
    #     traceback.print_exc()

if __name__ == "__main__":
     main()

    