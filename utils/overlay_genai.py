import cv2
import json
import re

def extract_json_sections(md_path):
    with open(md_path, "r") as f:
        content = f.read()
    # Find all JSON code blocks
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

def overlay_genai_video(
    video_path,
    md_path,
    output_path,
    fields=None,
    font_scale=0.6,
    color=(255,255,255),
    thickness=2,
    pos=(10, 30)
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
            "toolbox_placed_on_table",
            "num_chairs_stacked",
            "operator_holding",
            "gaze_target",
            "current_target_object",
            "current_phase",
            "Identified Key Objects",
            "Expected Immediate Next Action"
        ]

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
        json_idx = min(json_idx, max_json_idx)
        info = frame_info.get(json_idx)
        if info:
            lines = []
            for field in fields:
                if field in info:
                    lines.append(f"{field}: {info[field]}")
            frame = overlay_text(frame, lines, pos=pos, font_scale=font_scale, color=color, thickness=thickness)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Overlay video saved to {output_path}")

# Example usage:
# overlay_genai_video(
#     "/path/to/video.mp4",
#     "/path/to/history.md",
#     "/path/to/output_overlay.mp4",
#     fields=["gaze_target", "operator_holding"]
# )


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
    main()