import cv2
import json
import re
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.motionutils import get_end_effector_velocities 


def extract_json_sections(md_path):
    # If file is .json, load as JSON array
    if md_path.endswith('.json'):
        with open(md_path, "r") as f:
            data = json.load(f)
        frame_info = {}
        for idx, entry in enumerate(data):
            # Use "time" if available, else use index
            frame_idx = int(float(entry.get("time", idx)))
            frame_info[frame_idx] = entry
        return frame_info
    # Otherwise, treat as markdown with code blocks
    with open(md_path, "r") as f:
        content = f.read()
    matches = re.findall(r"```json\n(.*?)\n```", content, re.DOTALL)
    frame_info = {}
    for block in matches:
        try:
            data = json.loads(block)
            frame_idx = int(float(data.get("time", len(frame_info))))
            frame_info[frame_idx] = data
        except Exception as e:
            print(f"Failed to parse block: {e}")
    return frame_info

def overlay_text(frame, text_lines, pos=(10, 30), font_scale=0.7, color=(0,255,0), thickness=2):
    y = pos[1]
    for line in text_lines:
        cv2.putText(frame, line, (pos[0], y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        y += int(30 * font_scale)
    return frame


def overlay_genai_video_gt(
    video_path,
    md_path,
    output_path,
    fields=None,
    font_scale=0.6,
    color=(255,255,255),
    thickness=2,
    pos=(50, 100)
):
    """
    Overlays selected fields from JSON blocks in a markdown file onto a video.

    Args:
        video_path (str): Path to the input video.
        md_path (str): Path to the markdown file with JSON code blocks.
        output_path (str): Path to save the output video.
        fields (list of str): List of JSON fields to overlay. If None, uses a default set.
        font_scale (float): Font scale for overlay text.
        color (tuple): Text color (B, G, R).
        thickness (int): Text thickness.
        pos (tuple): Starting position (x, y) for text.
    """
    if fields is None:
        fields = [
            "time",
            "Expected Immediate Next Action"
        ]


    # Load JSON as a list of dicts
    with open(md_path, "r") as f:
        data = json.load(f)

    # Sort entries by frame_number
    entries = [entry for entry in data if "frame_number" in entry]
    entries.sort(key=lambda x: x["frame_number"])

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    entry_idx = 0
    num_entries = len(entries)

    for current_frame in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        # Advance entry_idx to the closest frame_number <= current_frame
        while entry_idx + 1 < num_entries and entries[entry_idx + 1]["frame_number"] <= current_frame:
            entry_idx += 1

        info = entries[entry_idx] if num_entries > 0 and entries[entry_idx]["frame_number"] <= current_frame else None
        lines = []
        if info:
            for field in fields:
                value = info.get(field, None)
                if value is not None:
                    # Format lists nicely
                    if isinstance(value, list):
                        lines.append(f"{field}: {', '.join(str(v) for v in value)}")
                    else:
                        lines.append(f"{field}: {value}")
        frame = overlay_text(frame, lines, pos=pos, font_scale=font_scale, color=color, thickness=thickness)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Overlay video saved to {output_path}")


def overlay_genai_video_phase2(
    video_path,
    md_path,
    output_path,
    fields=None,
    font_scale=0.6,
    color=(255,255,255),
    thickness=2,
    pos=(50, 100)
):
    """
    Overlays selected fields from JSON blocks in a markdown file onto a video.

    Args:
        video_path (str): Path to the input video.
        md_path (str): Path to the markdown file with JSON code blocks.
        output_path (str): Path to save the output video.
        fields (list of str): List of JSON fields to overlay. If None, uses a default set.
        font_scale (float): Font scale for overlay text.
        color (tuple): Text color (B, G, R).
        thickness (int): Text thickness.
        pos (tuple): Starting position (x, y) for text.
    """
    if fields is None:
        fields = [
            "time",
            "Expected Immediate Next Action"
        ]

    frame_info = extract_json_sections(md_path)

    # Build a sorted list of prediction frames
    prediction_frames = sorted([info.get("prediction_frame") for info in frame_info.values() if "prediction_frame" in info])
    prediction_idx = 0

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_json_idx = max(frame_info.keys()) if frame_info else 0

    for current_frame in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        # Find the current prediction_frame to use
        while (prediction_idx < len(prediction_frames) and current_frame > prediction_frames[prediction_idx]):
            prediction_idx += 1

        if prediction_idx < len(prediction_frames):
            pred_frame = prediction_frames[prediction_idx]
            if current_frame <= pred_frame:
                # Find the corresponding info
                info = None
                for v in frame_info.values():
                    if v.get("prediction_frame") == pred_frame:
                        info = v
                        break
                if info:
                    # Print the prediction result for this frame
                    print(f"Frame {current_frame}: Prediction for frame {pred_frame}: {info.get('predicted_positions', {})}")
                    lines = []
                    for field in fields:
                        if field in info:
                            lines.append(f"{field}: {info[field]}")
                        elif "predicted_positions" in info and field in info["predicted_positions"]:
                            lines.append(f"{field}: {info['predicted_positions'][field]}")
                    frame = overlay_text(frame, lines, pos=pos, font_scale=font_scale, color=color, thickness=thickness)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Overlay video saved to {output_path}")

def overlay_velocities_video(
    video_path,
    output_path,
    font_scale=0.5,
    color=(0, 255, 0), # Green for velocities
    thickness=1,
    pos=(10, 30)
):
    """
    Overlays motion velocities from HumanML3D data onto a video.
    The motion data file is assumed to be in ../data/humanml3d/ 
    and named identically to the video file (excluding extension), with a .pt extension.
    """
    velocity_scale = 100.0 # Scale for velocities
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    video_file_name_without_ext = os.path.splitext(os.path.basename(video_path))[0]
    motion_data_available = False
    total_motion_frames = 0

    try:
        # Check availability and get total motion frames
        _, _, _, _, _, _, _, total_motion_frames = get_end_effector_velocities(video_file_name_without_ext, 0)
        motion_data_available = True
        print(f"Motion data found for {video_file_name_without_ext}.pt with {total_motion_frames} frames.")
    except FileNotFoundError:
        print(f"Motion file {video_file_name_without_ext}.pt not found. Cannot overlay velocities.")
        cap.release() # Release video capture if motion data is essential and not found
        return
    except ValueError as e:
        print(f"Error reading initial motion data from {video_file_name_without_ext}.pt: {e}. Cannot overlay velocities.")
        cap.release()
        return
    except Exception as e:
        print(f"Unexpected error initializing motion data for {video_file_name_without_ext}.pt: {e}. Cannot overlay velocities.")
        cap.release()
        return

    for current_frame_idx in tqdm(range(total_video_frames), desc="Processing velocity overlay"):
        ret, frame = cap.read()
        if not ret:
            break

        lines = []
        if motion_data_available and current_frame_idx < total_motion_frames:
            try:
                (
                    root_ang_vel_y, root_lin_vel_x, root_lin_vel_z,
                    lf_v, rf_v, lh_v, rh_v, _ 
                ) = get_end_effector_velocities(video_file_name_without_ext, current_frame_idx)

                lines.append(f"Frame: {current_frame_idx}")
                lines.append(f"RootAngVelY: {velocity_scale * root_ang_vel_y:.2f}")
                lines.append(f"RootLinVelX: {velocity_scale * root_lin_vel_x:.2f}")
                lines.append(f"RootLinVelZ: {velocity_scale * root_lin_vel_z:.2f}")
                lines.append(f"LFootVel: {velocity_scale * lf_v:.2f}")
                lines.append(f"RFootVel: {velocity_scale * rf_v:.2f}")
                lines.append(f"LHandVel: {velocity_scale * lh_v:.2f}")
                lines.append(f"RHandVel: {velocity_scale * rh_v:.2f}")
            except ValueError as e: # If a specific frame fails within motion data
                if current_frame_idx % 100 == 0: # Log occasionally
                     print(f"Value error for motion data at frame {current_frame_idx}: {e}")
            except Exception as e: # Catch any other unexpected error during per-frame fetch
                if current_frame_idx % 100 == 0: # Log occasionally
                    print(f"Unexpected error fetching motion data for frame {current_frame_idx}: {e}")
        
        if lines:
            frame = overlay_text(frame, lines, pos=pos, font_scale=font_scale, color=color, thickness=thickness)
        
        out.write(frame)

    cap.release()
    out.release()
    print(f"Velocity overlay video saved to {output_path}")

def overlay_frame_numbers(video_path, output_path):
    """
    Overlays frame numbers on a video.
    
    Args:
        video_path (str): Path to the input video.
        output_path (str): Path to save the output video with frame numbers.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for current_frame in tqdm(range(total_frames), desc="Processing frame numbers"):
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, f"Frame: {current_frame}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255,255,255), 1, cv2.LINE_AA)
        out.write(frame)

    cap.release()
    out.release()
    print(f"Frame number overlay video saved to {output_path}")

def main():
    video_path = "/home/mani/Central/Cooking1/Stack/cam2_cr.mp4"
    md_path = "/home/mani/CLoSD/closd/IntentNet/history.md"
    output_path = "/home/mani/CLoSD/closd/IntentNet/cam2_cr_overlay.mp4"

    frame_info = extract_json_sections(md_path)

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_json_idx = max(frame_info.keys()) if frame_info else 0

    for current_frame in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        json_idx = current_frame // 30
        # Clamp to the last available json entry if video is longer than json
        json_idx = min(json_idx, max_json_idx)
        info = frame_info.get(json_idx)
        if info:
            lines = []
            if "toolbox_placed_on_table" in info:
                lines.append(f"Toolbox on table: {info['toolbox_placed_on_table']}")
            if "num_chairs_stacked" in info:
                lines.append(f"Chairs stacked: {info['num_chairs_stacked']}")
            if "operator_holding" in info:
                lines.append(f"Holding: {info['operator_holding']}")
            if "gaze_target" in info:
                lines.append(f"Gaze target: {info['gaze_target']}")
            if "current_target_object" in info:
                lines.append(f"Target object: {info['current_target_object']}")
            if "current_phase" in info:
                lines.append(f"Phase: {info['current_phase']}")
            if "Identified Key Objects" in info:
                lines.append(f"Objects: {info['Identified Key Objects']}")
            if "Expected Immediate Next Action" in info:
                lines.append(f"Next action: {info['Expected Immediate Next Action']}")
            # if "Predicted Action Description" in info:
            #     lines.append(f"Prediction: {info['Predicted Action Description']}")
            # if "Predicted Target Location" in info:
            #     lines.append(f"Target loc: {info['Predicted Target Location']}")
            frame = overlay_text(frame, lines, pos=(10, 30), font_scale=0.6, color=(255,255,255), thickness=2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Overlay video saved to {output_path}")

if __name__ == "__main__":
    # main()

    # overlay_genai_video(
    #     "/home/mani/Central/HaVid/S02A08I21/GVHMR/front/preprocess/vitpose_video_overlay_frames.mp4",
    #     "/home/mani/Repos/hcdt/data/HAViD/phase2_icl_result_window_3.json",
    #     "/home/mani/Central/HaVid/S02A08I21/GVHMR/front/preprocess/vitpose_video_overlay_frames_predictions_window3.mp4",
    #     fields=["prediction_frame","reasoning","target_object"]
    # )


    overlay_genai_video_gt(
        "/home/mani/Central/Stack/exp2/cam01.mp4",
        "data/Stack/exp2_gt.json",
        "data/OverlayVids/cam01_gt.mp4",
        fields=["frame_number","steps_completed","steps_in_progress","steps_available"]
    )


    # overlay_genai_video(
    #     "/home/mani/Central/Cooking1/aria01_214-1.mp4",
    #     "data/OverlayVids/gemini-flash-image-gen-cooking-dag.fixed.json",
    #     "data/overlay-gemini-flash-image-gen-cooking-dag.mp4",
    #     fields=["timestamp", "last_observed_action","expected_immediate_next_action"]    
    # )
    # video_path_velocities="/home/mani/Central/HaVid/S01A02I01S1.mp4"
    # output_path_velocities = "/home/mani/Central/HaVid/S01A02I01S1_velocities_overlay.mp4"
    # overlay_velocities_video(video_path_velocities, output_path_velocities,font_scale=2)

    # video_path = "/home/mani/Central/HaVid/ExampleContext/front.mp4"
    # output_path = "/home/mani/Central/HaVid/ExampleContext/front_f.mp4"
    # video_path = "/home/mani/Central/HaVid/S02A08I21/GVHMR/front/preprocess/vitpose_video_overlay.mp4"
    # output_path = "/home/mani/Central/HaVid/S02A08I21/GVHMR/front/preprocess/vitpose_video_overlay_frames.mp4"
    
    # overlay_frame_numbers(video_path, output_path)









