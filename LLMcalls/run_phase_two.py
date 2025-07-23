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
import hashlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.textutils import get_ground_truth
from utils.imgutils import image_to_base64
from utils.textutils import extract_json_from_response,get_action_description_for_frame, get_keystep_for_frame
from utils.experiment_logger import ExperimentLogger
from utils.motionutils import get_hand_xy_positions, get_end_effector_velocities
from utils.overlay_genai import overlay_genai_video_gt
from eval.run_evaluation import do_evaluation

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
    Formats the conversation history and response with specific image identifiers,
    and optionally saves it to a file. Handles text-only messages safely.
    
    Args:
        contents (List[types.Content]): The list of Content objects (conversation history).
        response_text (str): The final generated response from the model.
        file_path (Optional[str]): The path to the file to save the output.
        
    Returns:
        str: A formatted string representation of the contents and response.
    """
    output = []
    
    # Process all content objects, including the final user message
    for content in contents:
        # The model's response to the final message is handled after the loop
        if content.role == "user" and content == contents[-1]:
            continue

        role = content.role.capitalize()
        text_parts = []
        attachment_parts = []

        for part in content.parts:
            if hasattr(part, 'text') and part.text:
                text_parts.append(part.text)
            
            # This check now safely handles cases where inline_data is None
            if hasattr(part, 'inline_data') and part.inline_data:
                mime_type = part.inline_data.mime_type
                data = part.inline_data.data

                if 'image/' in mime_type:
                    image_hash = hashlib.sha256(data).hexdigest()
                    identifier = f"[Image Attached: {mime_type}, sha256: {image_hash[:12]}]"
                    attachment_parts.append(identifier)
                elif 'json' in mime_type:
                    attachment_parts.append("[JSON Data Attached]")

        full_text = "".join(text_parts)
        if attachment_parts:
            full_text += "\n" + "\n".join(attachment_parts)
        
        output.append(f"--- {role} Turn ---\n{full_text.strip()}")

        if content.role == "model":
            output.append("-" * 20)

    # Handle the final user prompt
    final_user_content = contents[-1]
    final_user_text_parts = [part.text for part in final_user_content.parts if hasattr(part, 'text') and part.text]
    final_user_full_text = "".join(final_user_text_parts)
    
    final_attachment_parts = []
    for part in final_user_content.parts:
        # Apply the same safe check here to prevent the crash
        if hasattr(part, 'inline_data') and part.inline_data:
            mime_type = part.inline_data.mime_type
            data = part.inline_data.data
            if 'image/' in mime_type:
                image_hash = hashlib.sha256(data).hexdigest()
                identifier = f"[Image Attached: {mime_type}, sha256: {image_hash[:12]}]"
                final_attachment_parts.append(identifier)
            elif 'json' in mime_type:
                final_attachment_parts.append("[JSON Data Attached]")

    if final_attachment_parts:
        final_user_full_text += "\n" + "\n".join(final_attachment_parts)

    output.append(f"--- User Turn ---\n{final_user_full_text.strip()}")
    
    # Add the final model response
    output.append(f"--- Model Turn ---\n{response_text.strip()}")
    output.append("-" * 20)

    formatted_string = "\n\n".join(output)
    
    if file_path:
        try:
            # Ensure the directory exists before writing the file
            if os.path.dirname(file_path):
                 os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
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
    # time.sleep(10)

    
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
    if first_image_paths and cfg.exp.type in ["ICL", "RCWPS"]:
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
                        filename = os.path.basename(img_path)
                        
                        # Add filename as text before the image
                        first_user_parts.append(types.Part.from_text(text=f"\nImage filename: {filename}"))
                        first_user_parts.append(types.Part.from_bytes(
                            mime_type=mime_type,
                            data=image_bytes
                        ))
    
    # Add test images for Phase 2 experiments (or any non-ICL experiment)
    if encoded_test_images and cfg.exp.type not in ["ICL", "RCWPS"]:
        for i, encoded_image in enumerate(encoded_test_images):
            if encoded_image:
                image_bytes = base64.b64decode(encoded_image)
                mime_type = get_image_mime_type(second_image_paths[i])
                first_user_parts.append(types.Part.from_bytes(
                    mime_type=mime_type,
                    data=image_bytes
                ))
        
    contents.append(types.Content(role="user", parts=first_user_parts))
    
    # ICL-specific conversation flow
    # Turn 2: Model acknowledges the examples
    if cfg.exp.type in ["ICL", "RCWPS"]:
        # model_reponse="Okay, I have analyzed the example video and supporting documents. I am ready for your questions about the test video."
        if cfg.exp.type == "RCWPS":
            model_reponse = cfg.model_response.v2
        elif cfg.exp.type == "ICL":
            model_reponse = cfg.model_response.v1

        contents.append(types.Content(role="model", parts=[types.Part.from_text(text=model_reponse)]))

        # Turn 3: User provides the next prompt and test images
        second_user_parts = [types.Part.from_text(text=next_query_prompt)]
        for i, encoded_image in enumerate(encoded_test_images):
            if encoded_image:
                image_bytes = base64.b64decode(encoded_image)
                img_path = second_image_paths[i]
                mime_type = get_image_mime_type(img_path)
                filename = os.path.basename(img_path)
                
                # Add filename as text before the image
                second_user_parts.append(types.Part.from_text(text=f"\nImage filename: {filename}"))
                second_user_parts.append(types.Part.from_bytes(
                    mime_type=mime_type,
                    data=image_bytes
                ))

        contents.append(types.Content(role="user", parts=second_user_parts))


    generate_content_config = types.GenerateContentConfig(
        thinking_config = types.ThinkingConfig(
            thinking_budget=512,), #Is this a hyperparameter?
        response_mime_type="text/plain",
        temperature=0.5,  
        top_p=0.95,        
        top_k=30 
        # response_mime_type="application/json",

    )
    start_time = time.time()
    full_response = []
    retries = 5
    for attempt in range(retries):
        try:
            # Reset full_response for each attempt
            full_response = []
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_content_config,
            ):
                if chunk.text is not None: # Check if chunk.text is not None
                    print(chunk.text, end="")
                    full_response.append(chunk.text)
            # If the loop completes without an exception, break out of the retry loop
            break
        except genai.errors.ClientError as e:
            print(f"\nClient error: {e}. Waiting to retry... (Attempt {attempt + 1}/{retries})")
            if attempt < retries - 1:
                time.sleep(15*attempt)  # Wait before retrying
            else:
                print("Max retries reached. Failing.")
                raise
        except genai.errors.ServerError as e:
            # if e.status_code == 500:
            #     print(f"\nServer internal error (500): {e}. Waiting to retry... (Attempt {attempt + 1}/{retries})")
            if attempt < retries - 1:
                time.sleep(attempt)  # Wait before retrying
            else:
                print("Max retries reached. Failing.")
                raise


    response_text = "".join(full_response)
    
    # End timing and log if logger is provided
    end_time = time.time()
    generation_duration = end_time - start_time
    
    if logger:
        # Log the full conversation to a file
        conversation_log_path = os.path.join(logger.output_dir, 'conversations', cfg.case_study, cfg.exp.type, cfg.model, f'frame_{frame_number}_conversation.log')
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

    # Initialize experiment logger 
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
    
    if cfg.exp.use_ego:
        test_ego_frame_files = sorted([f for f in os.listdir(cfg.exp.test_ego_frames) if f.endswith(('.png', '.jpg', '.jpeg'))])
        example_ego_frame_files = sorted([f for f in os.listdir(cfg.exp.example_ego_frames) if f.endswith(('.png', '.jpg', '.jpeg'))])
    # Process frames with 0.2 second intervals (every 3 frames at 15fps)
    frame_interval = int(0.2 * cfg.exp.fps)  # 0.2 seconds * 15 fps = 3 frames
    num_input_frames = 10
    
    all_responses_data = []
    script_dir = os.path.dirname(__file__)
    dag_file_path = os.path.join(script_dir, '..', 'data', cfg.case_study, 'dag.json')

    # Remove ground truth file loading - we'll use get_hand_xy_positions instead
    
    try:
        # Define the frame numbers to process dynamically
        fps = cfg.exp.fps
        start_frame = 1 + fps * 2
        if cfg.max_frames > 0:
            max_frame = cfg.max_frames
        else:
            max_frame = len(test_frame_files) - 2 * fps
        frame_increment = cfg.phase_two_increment
        frame_numbers = [start_frame + i * frame_increment for i in range((max_frame - start_frame) // frame_increment + 1)] #Usman: Changing casually to 240 for quick test
        # frame_numbers = [61, 301, 541, 781] # test

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
                if cfg.exp.use_keystep:
                    example_task_desc= get_keystep_for_frame(example_frame_num, cfg.exp.test_keystep_filename)
                else:
                    example_task_desc = get_action_description_for_frame(example_frame_num, cfg.exp.test_gt_filename, dag_file_path)
                example_prompt_text = prompt_template.format(task_description_string=example_task_desc)
                
                if cfg.exp.attach_drawing:
                    example_input_frames = [cfg.exp.drawing]
                    example_prompt_text += f"\n\n{cfg.exp.drawing_prompt}\n\n"

                if cfg.use_gaze:
                    example_prompt_text += f"\n\n{cfg.gaze_prompt}\n\n"

                if cfg.exp.use_ego:
                    example_prompt_text += f"\n\n{cfg.exp.ego_prompt}\n\n"

                example_hand_pos_data = []
                example_end_frame_idx = example_frame_num - 1
                for j in range(num_input_frames):
                    frame_idx = example_end_frame_idx - (num_input_frames - 1 - j) * frame_interval
                    if frame_idx >= 0 and frame_idx < len(test_frame_files):
                        example_input_frames.append(os.path.join(cfg.exp.test_vitpose_frames, test_frame_files[frame_idx]))
                        if cfg.use_gaze:
                            example_input_frames.append(os.path.join(cfg.exp.test_gazelle_output_dir, test_gaze_frame_files[frame_idx]))
                        if cfg.exp.use_ego:
                            example_input_frames.append(os.path.join(cfg.exp.test_ego_frames, test_ego_frame_files[frame_idx]))
                        try:
                            left_hand_x, left_hand_y, right_hand_x, right_hand_y = get_hand_xy_positions(cfg.exp.test_vitpose, frame_idx, frame_width=cfg.exp.frame_width, frame_height=cfg.exp.frame_height)
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
                prediction_frame = example_frame_num + cfg.prediction_delay_seconds * cfg.exp.fps
                try:
                    # Use get_hand_xy_positions to get actual hand positions for the prediction frame
                    pred_frame_idx = prediction_frame - 1  # Convert to 0-based index
                    left_hand_x, left_hand_y, right_hand_x, right_hand_y = get_hand_xy_positions(
                        cfg.exp.test_vitpose, pred_frame_idx, 
                        frame_width=cfg.exp.frame_width, frame_height=cfg.exp.frame_height
                    )
                    actual_hand_positions = {
                        "left_hand_x": left_hand_x,
                        "left_hand_y": left_hand_y,
                        "right_hand_x": right_hand_x,
                        "right_hand_y": right_hand_y
                    }
                    model_response = {"predicted_hand_positions": actual_hand_positions}
                    parts_model_response = [types.Part.from_text(text=json.dumps(model_response))]
                    contents.append(types.Content(role="model", parts=parts_model_response))
                except Exception as e:
                    print(f"Warning: Could not get hand positions for example prediction frame {prediction_frame}: {e}")
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
                        left_hand_x, left_hand_y, right_hand_x, right_hand_y = get_hand_xy_positions(cfg.exp.test_vitpose, frame_idx, frame_width=cfg.exp.frame_width, frame_height=cfg.exp.frame_height)
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
                    prediction_frame = frame_num + cfg.prediction_delay_seconds * cfg.exp.fps
                    
                    # Get actual hand positions for the prediction frame using get_hand_xy_positions
                    actual_positions = {}
                    try:
                        pred_frame_idx = prediction_frame - 1  # Convert to 0-based index
                        left_hand_x, left_hand_y, right_hand_x, right_hand_y = get_hand_xy_positions(
                            cfg.exp.test_vitpose, pred_frame_idx, 
                            frame_width=cfg.exp.frame_width, frame_height=cfg.exp.frame_height
                        )
                        actual_positions = {
                            "left_hand_x": left_hand_x,
                            "left_hand_y": left_hand_y,
                            "right_hand_x": right_hand_x,
                            "right_hand_y": right_hand_y
                        }
                    except Exception as e:
                        print(f"Warning: Could not get actual hand positions for prediction frame {prediction_frame}: {e}")

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
    result_store_path = os.path.join(
        script_dir, 
        '..', 
        'logs', 
        cfg.exp.type + '_' + cfg.case_study + '_' + cfg.model + ('_use_gaze' if cfg.use_gaze else '_no_gaze') + '_result.json'
    )
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

    if cfg.exp.use_ego:
        prompt_text += f"\n\n{cfg.exp.ego_prompt}\n\n"
        
    # Accept both .png, .jpg, .jpeg files for examples
    example_frame_files = sorted([
        f for f in os.listdir(cfg.exp.example_image_dir)
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ])
    for i in range(0, len(example_frame_files), cfg.exp.example_frame_step):
        first_image_paths.append(os.path.join(cfg.exp.example_image_dir, example_frame_files[i]))

    # Accept both .png, .jpg, .jpeg files for test frames
    if cfg.total_frames_to_process == 'all':
        total_frames_to_process = len([
            f for f in os.listdir(cfg.exp.test_image_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
    all_responses_data = []
    
    current_response = None
    frame_num = cfg.start_frame

    if cfg.use_gaze:
        test_gaze_frame_files = sorted([f for f in os.listdir(cfg.exp.test_gazelle_output_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        example_gaze_frame_files = sorted([f for f in os.listdir(cfg.exp.example_gazelle_output_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if cfg.exp.use_ego:
        test_ego_frame_files = sorted([f for f in os.listdir(cfg.exp.test_ego_frames) if f.endswith(('.png', '.jpg', '.jpeg'))])
        example_ego_frame_files = sorted([f for f in os.listdir(cfg.exp.example_ego_frames) if f.endswith(('.png', '.jpg', '.jpeg'))])

    

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
            # Accept both .png, .jpg, .jpeg files for test frames
            test_frame_files = sorted([
                f for f in os.listdir(cfg.exp.test_image_dir)
                if f.endswith(('.png', '.jpg', '.jpeg'))
            ])
            for i in range(0, frame_num + 1, cfg.exp.test_frame_step):
                if i < len(test_frame_files):
                    second_image_paths.append(os.path.join(cfg.exp.test_image_dir, test_frame_files[i]))
                    if cfg.use_gaze:
                        second_image_paths.append(os.path.join(cfg.exp.test_gazelle_output_dir, test_gaze_frame_files[i]))
                    if cfg.exp.use_ego:
                        second_image_paths.append(os.path.join(cfg.exp.test_ego_frames, test_ego_frame_files[i]))


            # if cfg.use_openai:
            #     current_response = generate_openAI(cfg, prompt_text, second_image_paths, first_image_paths, second_prompt_text, logger, frame_num, json_data=json_data)
            # else:
            
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
        print(f"\n✅ Experiment finished. Results saved to {result_store_path}")
        logger.end_experiment()
        if cfg.overlay_results:
            output_video_path = os.path.splitext(result_store_path)[0] + '.mp4'
            overlay_genai_video_gt(
                video_path=cfg.exp.test_video_file_path,
                md_path=result_store_path,
                output_path=output_video_path
            )
        if cfg.run_evaluation:
            print("\nRunning evaluation...")
            eval_output_dir = os.path.join(os.path.dirname(result_store_path), 'evaluation_results', os.path.basename(os.path.splitext(result_store_path)[0]))
            do_evaluation(
                icl_file=result_store_path,
                gt_file=cfg.exp.test_gt_filename,
                output_dir=eval_output_dir
            )

def runRCWPS(cfg: DictConfig):
    
    # Initialize experiment logger
    output_dir = os.getcwd()
    output_dir = os.path.join(output_dir, 'logs/logger')
    logger = ExperimentLogger(output_dir=output_dir)
    
    # Start experiment with notes
    experiment_notes = f"RCWPS experiment on {cfg.case_study} dataset"
    experiment_id = logger.start_experiment(cfg, experiment_notes)
    
    script_dir = os.path.dirname(__file__)
    dag_file_path = os.path.join(script_dir, '..', 'data', cfg.case_study, 'dag.json')
    state_file_path = os.path.join(script_dir, '..', 'data', cfg.case_study, 'state.json')
    # example_gt_path=  os.path.join(script_dir, '..', 'data', case_study, 'S02A08I21_gt.json')
    result_store_path = os.path.join(
        script_dir, 
        '..', 
        'logs', 
        cfg.exp.type + '_' + cfg.case_study + '_' + cfg.model + ('_use_gaze' if cfg.use_gaze else '_no_gaze') + ('_use_gt' if cfg.use_ground_truth else '') + '_result.json'
    )
    
    with open(dag_file_path, 'r') as f:
            task_graph_string = f.read()
    with open(state_file_path, 'r') as f:
            state_schema_string = f.read()
    with open(cfg.exp.example_gt_filename, 'r') as f:
            example_gt_string = f.read()


    
    prompt_template = cfg.prompts.RCWPS

  
        
    if cfg.total_frames_to_process == 'all':
        total_frames_to_process = len([
            f for f in os.listdir(cfg.exp.test_image_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
    all_responses_data = []
    
    current_response = None
    frame_num = cfg.start_frame

    if cfg.use_gaze:
        test_gaze_frame_files = sorted([f for f in os.listdir(cfg.exp.test_gazelle_output_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        example_gaze_frame_files = sorted([f for f in os.listdir(cfg.exp.example_gazelle_output_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if cfg.exp.use_ego:
        test_ego_frame_files = sorted([f for f in os.listdir(cfg.exp.test_ego_frames) if f.endswith(('.png', '.jpg', '.jpeg'))])
        example_ego_frame_files = sorted([f for f in os.listdir(cfg.exp.example_ego_frames) if f.endswith(('.png', '.jpg', '.jpeg'))])

    

    try:
        while frame_num < total_frames_to_process:
            if cfg.end_frame > 0 and frame_num >= cfg.end_frame:
                break

             # Calculate the starting frame to ensure maximum 10 frames
            max_frames = 10
            total_possible_frames = len(range(0, frame_num, cfg.exp.test_frame_step))
            if total_possible_frames <= max_frames:
                start_frame = 0
                previous_state = get_ground_truth(1, cfg.exp.test_gt_filename)
            else:
                # Start from a frame that gives us exactly 10 frames
                start_frame = frame_num - (max_frames - 1) * cfg.exp.test_frame_step

            if start_frame > 0:
                if cfg.use_ground_truth:
                    previous_state = get_ground_truth(start_frame+1, cfg.exp.test_gt_filename)
                else:
                    previous_state = get_ground_truth(start_frame+1, result_store_path)
                
            prompt_text = prompt_template.format(
                task_graph_string=task_graph_string,
                state_schema_string=state_schema_string,
                number_of_frames_attached=cfg.max_attached_frames,
                fps=cfg.exp.fps,
                test_frame_step=cfg.exp.test_frame_step,
                test_frame_step_seconds=cfg.exp.test_frame_step / cfg.exp.fps,
                previous_state_string = previous_state
            )

            second_prompt_template = cfg.second_prompt.v1
            second_prompt_text = second_prompt_template.format(                
                fps=cfg.exp.fps,
                frame_num=frame_num,
                last_frame_time=f"{frame_num / cfg.exp.fps:.2f}"
            )
            json_data = None  # No JSON data needed for version1 prompt
            
            first_image_paths = []

            if cfg.exp.attach_drawing:
                first_image_paths = [cfg.exp.drawing]        
                prompt_text += f"\n\n{cfg.exp.drawing_prompt}\n\n"

            if cfg.exp.use_ego:
                prompt_text += f"\n\n{cfg.exp.ego_prompt}\n\n"

            second_image_paths = []
            # Accept both .png, .jpg, .jpeg files for test frames
            test_frame_files = sorted([
                f for f in os.listdir(cfg.exp.test_image_dir)
                if f.endswith(('.png', '.jpg', '.jpeg'))
            ])
            
           
                
            for i in range(start_frame, frame_num + 1, cfg.exp.test_frame_step):
                if i < len(test_frame_files):
                    second_image_paths.append(os.path.join(cfg.exp.test_image_dir, test_frame_files[i]))
                    if cfg.use_gaze:
                        second_image_paths.append(os.path.join(cfg.exp.test_gazelle_output_dir, test_gaze_frame_files[i]))
                    if cfg.exp.use_ego:
                        second_image_paths.append(os.path.join(cfg.exp.test_ego_frames, test_ego_frame_files[i]))


            
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
        print(f"\n✅ Experiment finished. Results saved to {result_store_path}")
        if cfg.overlay_results:
            output_video_path = os.path.splitext(result_store_path)[0] + '.mp4'
            overlay_genai_video_gt(
                video_path=cfg.exp.test_video_file_path,
                md_path=result_store_path,
                output_path=output_video_path
            )
        if cfg.run_evaluation:
            print("\nRunning evaluation...")
            eval_output_dir = os.path.join(os.path.dirname(result_store_path), 'evaluation_results', os.path.basename(os.path.splitext(result_store_path)[0]))
            do_evaluation(
                icl_file=result_store_path,
                gt_file=cfg.exp.test_gt_filename,
                output_dir=eval_output_dir
            )

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))  # Print the entire config for debugging
    if cfg.exp.type=="ICL":
        runICL_HI(cfg)
    elif cfg.exp.type == "phase2":
        runPhase2(cfg)
    elif cfg.exp.type == "RCWPS":
        runRCWPS(cfg)

if __name__ == "__main__":
     main()

