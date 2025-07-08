import os
import subprocess
import glob
import math

def create_video_with_pauses(    
    input_frames_dir="frames",
    output_video_path="output_video_with_pauses.mp4",
    base_fps=15,
    pause_duration_seconds=1,
    initial_pause_frame=106,
    subsequent_pause_interval=60
):
    """
    Combines a sequence of image frames into a video with specified pauses.

    Args:
        input_frames_dir (str): Directory containing the image frames.
                                 Frames are expected to be named like frame_0001.png, frame_0002.png, etc.
        output_video_path (str): Path for the output video file (e.g., "output_video.mp4").
        base_fps (int): The base frame rate for the video.
        pause_duration_seconds (int): The duration of the pause (in seconds) at specified frames.
        initial_pause_frame (int): The first frame number where a pause should occur.
        subsequent_pause_interval (int): The interval (in frames) for subsequent pauses
                                         after the initial_pause_frame.
    """

    # --- Step 1: Validate input directory and find frames ---
    if not os.path.isdir(input_frames_dir):
        print(f"Error: Input frames directory '{input_frames_dir}' not found.")
        print("Please ensure your frames are in a folder named 'frames' (or specify a different folder).")
        print("Example frame names: frame_0001.png, frame_0002.png, etc.")
        return

    # Find all frame files and sort them numerically
    # Using glob to find files matching the pattern, then sorting to ensure correct order
    frame_files = sorted(glob.glob(os.path.join(input_frames_dir, "frame_*.png")))
    if not frame_files:
        print(f"Error: No frame files found in '{input_frames_dir}'.")
        print("Please ensure your frames are named like 'frame_0001.png', 'frame_0002.png', etc.")
        return

    num_frames = len(frame_files)
    if num_frames == 0:
        print("No frames found to process. Exiting.")
        return

    print(f"Found {num_frames} frames in '{input_frames_dir}'.")

    # --- Step 2: Calculate frame durations and create FFmpeg concat list ---
    # Normal duration for a single frame at the base FPS
    normal_frame_duration = 1.0 / base_fps

    # Path for the temporary FFmpeg concat list file
    concat_list_path = "ffmpeg_input_list.txt"

    with open(concat_list_path, "w") as f:
        for i, frame_path in enumerate(frame_files):
            # Frame numbers are 1-based for user understanding (e.g., frame 106)
            frame_number = i + 1

            current_duration = normal_frame_duration

            # Check if the current frame is a designated pause frame
            is_pause_frame = False
            if frame_number == initial_pause_frame:
                is_pause_frame = True
            elif frame_number > initial_pause_frame and \
                 (frame_number - initial_pause_frame) % subsequent_pause_interval == 0:
                is_pause_frame = True

            # If it's a pause frame, add the extra pause duration
            if is_pause_frame:
                current_duration += pause_duration_seconds
                print(f"  Adding {pause_duration_seconds}s pause at frame {frame_number} ({os.path.basename(frame_path)})")

            # Write the file path and its effective duration to the concat list
            # Using os.path.abspath ensures FFmpeg can find the files regardless of the current working directory
            f.write(f"file '{os.path.abspath(frame_path)}'\n")
            f.write(f"duration {current_duration}\n")

        # IMPORTANT: The concat demuxer requires the last file to have a 'duration' line
        # to ensure it's fully included in the output. We re-add the last frame's path
        # and its normal duration to finalize the stream.
        f.write(f"file '{os.path.abspath(frame_files[-1])}'\n")
        f.write(f"duration {normal_frame_duration}\n")

    print(f"Generated FFmpeg concat list: '{concat_list_path}'")

    # --- Step 3: Execute FFmpeg command ---
    ffmpeg_command = [
        "ffmpeg",
        "-f", "concat",          # Use the concat demuxer
        "-safe", "0",            # Allows paths outside the current directory (needed for abspath)
        "-i", concat_list_path,  # Input is the generated list file
        "-c:v", "libx264",       # Video codec: H.264 (widely compatible)
        "-pix_fmt", "yuv420p",   # Pixel format: YUV 4:2:0 (standard for broad compatibility)
        "-r", str(base_fps),     # Output video frame rate (FFmpeg will duplicate/drop frames to match durations)
        "-movflags", "+faststart", # Optimize MP4 for web streaming (metadata at start)
        output_video_path
    ]

    print(f"\nExecuting FFmpeg command:\n{' '.join(ffmpeg_command)}\n")

    try:
        # Run FFmpeg as a subprocess
        # check=True will raise CalledProcessError if FFmpeg returns a non-zero exit code
        # stdout and stderr are captured for better error reporting
        process = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
        print(f"Video '{output_video_path}' created successfully!")
        if process.stdout:
            print("FFmpeg STDOUT:\n", process.stdout)
        if process.stderr:
            print("FFmpeg STDERR:\n", process.stderr)

    except subprocess.CalledProcessError as e:
        print(f"Error during FFmpeg execution (Exit Code: {e.returncode}):")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
    except FileNotFoundError:
        print("\nError: FFmpeg not found.")
        print("Please ensure FFmpeg is installed on your system and accessible via your system's PATH.")
        print("You can download FFmpeg from https://ffmpeg.org/download.html")
    finally:
        # --- Step 4: Clean up temporary file ---
        if os.path.exists(concat_list_path):
            os.remove(concat_list_path)
            print(f"\nCleaned up temporary file: '{concat_list_path}'")


if __name__ == "__main__":
    create_video_with_pauses(
        input_frames_dir="/home/mani/Central/HaVid/S02A08I21/GVHMR/front/preprocess/VitPose-overlay",
        output_video_path="/home/mani/Central/HaVid/S02A08I21/GVHMR/front/preprocess/my_custom_video.mp4",
        base_fps=15,
        pause_duration_seconds=1,
        initial_pause_frame=106,
        subsequent_pause_interval=60
    )