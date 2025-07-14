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
from utils.textutils import extract_json_from_response,get_action_description_for_frame
from utils.experiment_logger import ExperimentLogger
from utils.motionutils import get_hand_xy_positions, get_end_effector_velocities

from openai import OpenAI

def get_image_mime_type(file_path: str) -> str:
    """
    Determines the MIME type based on the file extension.
    
    Args:
        file_path (str): Path to the image file.
        
    Returns:
        str: The appropriate MIME type.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.jpg', '.jpeg']:
        return "image/jpeg"
    elif ext == '.png':
        return "image/png"
    else:
        # Default to PNG for unknown extensions
        return "image/png"

def output_contents(contents: List[types.Content], response_text: str, file_path: Optional[str] = None) -> str:
    """
    Formats the conversation history and response, and optionally saves it to a file.
    
    Args:
        contents (List[types.Content]): The list of Content objects (conversation history).
        response_text (str): The final generated response from the model.
        file_path (Optional[str]): The path to the file to save the output.
        
    Returns:
        str: A formatted string representation of the contents and response.
    """
    output = []
    for content in contents:
        role = content.role.capitalize()
        # The last user message is part of the history but its response isn't yet,
        # so we handle it separately after the loop.
        if content.role == "user" and content == contents[-1]:
            continue

        text_parts = []
        has_image = False
        has_json = False

        for part in content.parts:
            # Use hasattr to safely check for attributes
            if hasattr(part, 'text') and part.text:
                text_parts.append(part.text)
            if hasattr(part, 'mime_type'):
                if 'image/' in part.mime_type:
                    has_image = True
                elif 'json' in part.mime_type:
                    has_json = True
        
        full_text = "".join(text_parts)
        if has_image:
            full_text += "\n[Image Content Attached]"
        if has_json:
            full_text += "\n[JSON Data Attached]"
        
        output.append(f"--- {role} Turn ---\n{full_text.strip()}")

        # For ICL examples, the model's response is already in `contents`
        if content.role == "model":
            # The model's response text is already in full_text, 
            # so we just need to add the separator.
            output.append("-" * 20)


    # Handle the final user prompt and the model's response to it
    final_user_content = contents[-1]
    final_user_text_parts = [part.text for part in final_user_content.parts if hasattr(part, 'text') and part.text]
    final_user_full_text = "".join(final_user_text_parts)
    if any('image/' in part.mime_type for part in final_user_content.parts if hasattr(part, 'mime_type')):
        final_user_full_text += "\n[Image Content Attached]"
    if any('json' in part.mime_type for part in final_user_content.parts if hasattr(part, 'mime_type')):
        final_user_full_text += "\n[JSON Data Attached]"

    output.append(f"--- User Turn ---\n{final_user_full_text.strip()}")
    output.append(f"--- Model Turn ---\n{response_text.strip()}")
    output.append("-" * 20)

    formatted_string = "\n\n".join(output)
    
    if file_path:
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                f.write(formatted_string)
        except Exception as e:
            print(f"Error writing to log file {file_path}: {e}")

    return formatted_string

def generate(cfg: DictConfig,
    query_prompt: str,
    second_image_paths: Optional[List[str]] = None,   
    first_image_paths: Optional[List[str]] = None,  
    next_query_prompt: Optional[str] = None, 
    logger: ExperimentLogger = None,
    frame_number: int = 0,
    responses: Optional[List[str]] = None,
    json_data: Optional[List[str]] = None,
    contents: List[types.Content] = None):

    print_formatted = True  # Set to True to print formatted output

    encoded_test_images = []
    for img_path in second_image_paths:
        encoded_img = image_to_base64(img_path, target_largest_dimension=640)
        encoded_test_images.append(encoded_img) # Will store None if encoding failed
    
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )
    
    model=cfg.model

    start_time = time.time()
    
    # `contents` is now passed in, representing the conversation history.
    # The function will append the current turn's user message to it.
    if contents is None:
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

    # Add example images (for ICL experiments)
    if first_image_paths and cfg.exp.type == "ICL":
        if cfg.use_file_upload:
            for img_path in first_image_paths:
                uploaded_file = client.files.upload(file=img_path)
                mime_type = get_image_mime_type(img_path)
                first_user_parts.append(types.Part.from_uri(file_uri=uploaded_file.uri, mime_type=mime_type))  
        else:
            encoded_example_images = []
            for img_path in first_image_paths:
                encoded_img = image_to_base64(img_path, target_largest_dimension=640)
                encoded_example_images.append((encoded_img, img_path))

            if encoded_example_images:
                for encoded_image, img_path in encoded_example_images:
                    if encoded_image:
                        image_bytes = base64.b64decode(encoded_image)
                        mime_type = get_image_mime_type(img_path)
                        first_user_parts.append(types.Part.from_bytes(
                            mime_type=mime_type,
                            data=image_bytes
                        ))
    
    # Add test images for Phase 2 experiments (or any non-ICL experiment)
    if encoded_test_images and cfg.exp.type != "ICL":
        for i, encoded_image in enumerate(encoded_test_images):
            if encoded_image:
                image_bytes = base64.b64decode(encoded_image)
                mime_type = get_image_mime_type(second_image_paths[i])
                first_user_parts.append(types.Part.from_bytes(
                    mime_type=mime_type,
                    data=image_bytes
                ))
        
    contents.append(types.Content(role="user", parts=first_user_parts))
    
    # ICL-specific conversation flow is now handled in runPhase2
    generate_content_config = types.GenerateContentConfig(
        thinking_config = types.ThinkingConfig(
            thinking_budget=1,), #Is this a hyperparameter?
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
        # Log the full conversation to a file
        conversation_log_path = os.path.join(logger.output_dir, 'conversations', f'frame_{frame_number}_conversation.log')
        output_contents(contents, response_text, file_path=conversation_log_path)
        print(f"Full conversation saved to {conversation_log_path}")
        
        # Create combined prompt for token counting
        combined_prompt = f"{query_prompt}\n\n{next_query_prompt}"
        
        logger.log_generation(
            frame_number=frame_number,
            prompt=combined_prompt,
            response=response_text,
            duration=generation_duration,
            model=model,
            example_images_count=len(first_image_paths) if first_image_paths else 0,
            test_images_count=len(second_image_paths) if second_image_paths else 0,
            temperature=generate_content_config.temperature,
            top_p=generate_content_config.top_p,
            top_k=generate_content_config.top_k
        )
    
    return response_text


def runPhase2(cfg: DictConfig):
    """
    Runs the second phase of the experiment, which inputting 10 images 0.2 seconds apart with human pose overlay
    and left and right hand pixel locations, and asks the gemini model to predict left and right hand pixel locations
    two seconds after the last frame of the first phase. 
    and generating responses using the configured model.

    Args:
        cfg (DictConfig): The configuration object containing experiment settings.
    """

    # Initialize experiment logger (match run_ICL_experiments.py style)
    output_dir = os.getcwd()
    output_dir = os.path.join(output_dir, 'logs')
    logger = ExperimentLogger(output_dir=output_dir)

    # Start experiment with notes 
    experiment_notes = f"Phase 2 experiment: Hand position prediction using {cfg.model} on {cfg.case_study} dataset"
    experiment_id = logger.start_experiment(cfg, experiment_notes)

    # Build prompt
    prompt_template = cfg.prompts.phase_two

    # Get test image files (support both jpg and png)
    test_frame_files = sorted([f for f in os.listdir(cfg.exp.test_vitpose_frames) if f.endswith(('.png', '.jpg', '.jpeg'))])
    if cfg.use_gaze:
        test_gaze_frame_files = sorted([f for f in os.listdir(cfg.exp.test_gazelle_output_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        example_gaze_frame_files = sorted([f for f in os.listdir(cfg.exp.example_gazelle_output_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    # Process frames with 0.2 second intervals (every 3 frames at 15fps)
    frame_interval = int(0.2 * cfg.exp.fps)  # 0.2 seconds * 15 fps = 3 frames
    num_input_frames = 10
    
    all_responses_data = []
    script_dir = os.path.dirname(__file__)
    dag_file_path = os.path.join(script_dir, '..', 'data', cfg.case_study, 'dag.json')

    # Load ground truth data for examples
    with open(cfg.exp.test_phase2_ground_truth_file, 'r') as f:
        ground_truth_data = json.load(f)
    
    gt_map = {item['frame']: item for item in ground_truth_data}

    try:
        # Define the frame numbers to process dynamically
        fps = cfg.exp.fps
        start_frame = 1 + fps * 2
        max_frame = len(test_frame_files) - 2 * fps
        frame_numbers = [start_frame + i * 60 for i in range((max_frame - start_frame) // 60 + 1)]
        
        # Main loop for processing each frame that needs a prediction
        for i, frame_num in enumerate(frame_numbers):
            # Skip the frames that will only be used as examples
            if i < cfg.exp.num_examples:
                continue

            print(f"\n🔄 Processing prediction for frame {frame_num}...")
            
            # --- Start of Context Window Creation ---
            # Create a fresh conversation history for each prediction
            contents = []
            
            # Get the starting index for the examples
            start_example_index = i - cfg.exp.num_examples
            
            # Loop through the frames that will serve as examples
            for example_index in range(start_example_index, i):
                example_frame_num = frame_numbers[example_index]
                print(f"  -> Adding example from frame {example_frame_num}")

                # This block is similar to your original loop, but now it's for building examples
                example_input_frames = []
                example_task_desc = get_action_description_for_frame(example_frame_num, cfg.exp.test_gt_filename, dag_file_path)
                example_prompt_text = prompt_template.format(task_description_string=example_task_desc)
                
                if cfg.exp.attach_drawing:
                    example_input_frames = [cfg.exp.drawing]
                    example_prompt_text += f"\n\n{cfg.exp.drawing_prompt}\n\n"

                if cfg.use_gaze:
                    example_prompt_text += f"\n\n{cfg.gaze_prompt}\n\n"

                example_hand_pos_data = []
                example_end_frame_idx = example_frame_num - 1
                for j in range(num_input_frames):
                    frame_idx = example_end_frame_idx - (num_input_frames - 1 - j) * frame_interval
                    if frame_idx >= 0 and frame_idx < len(test_frame_files):
                        example_input_frames.append(os.path.join(cfg.exp.test_vitpose_frames, test_frame_files[frame_idx]))
                        if cfg.use_gaze:
                            example_input_frames.append(os.path.join(cfg.exp.test_gazelle_output_dir, test_gaze_frame_files[frame_idx]))
                        try:
                            left_hand_x, left_hand_y, right_hand_x, right_hand_y = get_hand_xy_positions(cfg.exp.test_vitpose, frame_idx)
                            (_, _, _, _, _, left_hand_vel, right_hand_vel, _) = get_end_effector_velocities(cfg.exp.test_humanml3d, frame_idx)
                            left_hand_vel *= 100
                            right_hand_vel *= 100
                            example_hand_pos_data.append({
                                'frame': frame_idx + 1, 'time_seconds': frame_idx / cfg.exp.fps,
                                'left_hand_x': left_hand_x, 'left_hand_y': left_hand_y,
                                'right_hand_x': right_hand_x, 'right_hand_y': right_hand_y,
                                'left_hand_velocity': left_hand_vel, 'right_hand_velocity': right_hand_vel
                            })
                        except Exception as e:
                            print(f"Error getting hand data for example frame {frame_idx}: {e}")
                
                example_prompt_text += "\n\nHand and Pose Data:\n" + json.dumps(example_hand_pos_data, indent=2)

                # Add the user part of the example to history
                user_parts = [types.Part.from_text(text=example_prompt_text)]
                encoded_images = [(image_to_base64(img, 640), img) for img in example_input_frames if img]
                for encoded_image, img_path in encoded_images:
                    if encoded_image:
                        image_bytes = base64.b64decode(encoded_image)
                        mime_type = get_image_mime_type(img_path)
                        user_parts.append(types.Part.from_bytes(mime_type=mime_type, data=image_bytes))
                contents.append(types.Content(role="user", parts=user_parts))

                # Add the model part (ground truth) of the example to history
                prediction_frame = example_frame_num + 2*cfg.exp.fps  # 2 seconds * 15 fps = 30 frames
                if prediction_frame in gt_map:
                    gt_entry = gt_map[prediction_frame]
                    model_response = {"predicted_hand_positions": gt_entry.get("actual_hand_positions", {})}
                    parts_model_response = [types.Part.from_text(text=json.dumps(model_response))]
                    contents.append(types.Content(role="model", parts=parts_model_response))
                else:
                    print(f"Warning: Ground truth for example prediction frame {prediction_frame} not found.")
            # --- End of Context Window Creation ---


            # Now, prepare the actual prompt for the current frame_num
            print(f"  => Preparing prompt for actual prediction at frame {frame_num}")
            input_frames = []
            task_description_string = get_action_description_for_frame(frame_num, cfg.exp.test_gt_filename, dag_file_path)
            prompt_text = prompt_template.format(task_description_string=task_description_string)
            
            if cfg.exp.attach_drawing:
                input_frames = [cfg.exp.drawing]
                prompt_text += f"\n\n{cfg.exp.drawing_prompt}\n\n"
            
            hand_positions_data = []
            end_frame_idx = frame_num - 1

            for j in range(num_input_frames):
                frame_idx = end_frame_idx - (num_input_frames - 1 - j) * frame_interval
                if frame_idx >= 0 and frame_idx < len(test_frame_files):
                    input_frames.append(os.path.join(cfg.exp.test_vitpose_frames, test_frame_files[frame_idx]))
                    try:
                        left_hand_x, left_hand_y, right_hand_x, right_hand_y = get_hand_xy_positions(cfg.exp.test_vitpose, frame_idx)
                        (_, _, _, _, _, left_hand_vel, right_hand_vel, _) = get_end_effector_velocities(cfg.exp.test_humanml3d, frame_idx)
                        left_hand_vel *= 100
                        right_hand_vel *= 100
                        hand_positions_data.append({
                            'frame': frame_idx + 1, 'time_seconds': frame_idx / cfg.exp.fps,
                            'left_hand_x': left_hand_x, 'left_hand_y': left_hand_y,
                            'right_hand_x': right_hand_x, 'right_hand_y': right_hand_y,
                            'left_hand_velocity': left_hand_vel, 'right_hand_velocity': right_hand_vel
                        })
                    except Exception as e:
                        print(f"Error getting hand data for frame {frame_idx}: {e}")
            
            prompt_text += "\n\nHand and Pose Data:\n" + json.dumps(hand_positions_data, indent=2)

            # Call the generate function with the freshly prepared conversation history
            response_text = generate(
                cfg=cfg,
                query_prompt=prompt_text,
                second_image_paths=input_frames,
                logger=logger,
                frame_number=frame_num,
                contents=contents  # Pass the newly created context
            )

            # Extract JSON from the response
            json_part = extract_json_from_response(response_text)
            
            if json_part:
                try:
                    response_data = json.loads(json_part)
                    prediction_frame = frame_num + 2*cfg.exp.fps  # 2 seconds * 15 fps = 30 frames
                    
                    # Get actual hand positions for the prediction frame for comparison
                    actual_positions = {}
                    if prediction_frame in gt_map:
                        actual_positions = gt_map[prediction_frame].get('actual_hand_positions', {})

                    all_responses_data.append({
                        "input_end_frame": frame_num,
                        "prediction_frame": prediction_frame,
                        "predicted_hand_positions": response_data.get('predicted_hand_positions', {}),
                        "actual_hand_positions": actual_positions,
                        "reasoning": response_data.get('reasoning', ''),
                        "target_object": response_data.get('target_object', '')
                    })
                except json.JSONDecodeError:
                    print(f"Error: Could not decode JSON from response for frame {frame_num}")
            else:
                print(f"Error: No JSON found in response for frame {frame_num}")

    except Exception as e:
        print(f"❌ Phase 2 experiment failed: {e}")
        raise
    finally:
        # Always end the experiment to save logs
        logger.end_experiment()
        # Save results to a file
        results_filename = f'phase2_icl_result_window_{cfg.exp.num_examples}.json'
        results_file_path = os.path.join(
            os.path.dirname(__file__), '..', 'data', cfg.case_study, results_filename
        )
        with open(results_file_path, 'w') as f:
            json.dump(all_responses_data, f, indent=4)
        
        print(f"\n✅ Phase 2 experiment finished. Results saved to {results_file_path}")

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    runPhase2(cfg)

if __name__ == "__main__":
     main()

