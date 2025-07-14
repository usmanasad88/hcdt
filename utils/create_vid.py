import os
import subprocess
import glob
import math

def create_video_with_pauses(
    input_frames_dir="frames",
    base_frames_dir=None,
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
                                 If base_frames_dir is provided, these are all pause frames.
                                 Otherwise, frames are expected to be named like frame_0001.png, etc.
        base_frames_dir (str, optional): Directory for base frames. If provided,
                                         input_frames_dir are pause frames. Defaults to None.
        output_video_path (str): Path for the output video file (e.g., "output_video.mp4").
        base_fps (int): The base frame rate for the video.
        pause_duration_seconds (int): The duration of the pause (in seconds) at specified frames.
        initial_pause_frame (int): The first frame number where a pause should occur.
                                   (Only used when base_frames_dir is None).
        subsequent_pause_interval (int): The interval (in frames) for subsequent pauses.
                                         (Only used when base_frames_dir is None).
    """

    # --- Step 1: Validate input directories and find frames ---
    if not os.path.isdir(input_frames_dir):
        print(f"Error: Input frames directory '{input_frames_dir}' not found.")
        return

    frame_files = []
    pause_frame_paths = set()

    if base_frames_dir:
        if not os.path.isdir(base_frames_dir):
            print(f"Error: Base frames directory '{base_frames_dir}' not found.")
            return
        print(f"Using base frames from '{base_frames_dir}' and pause frames from '{input_frames_dir}'.")
        base_frame_files = sorted(glob.glob(os.path.join(base_frames_dir, "frame_*.png")))
        pause_frame_files = sorted(glob.glob(os.path.join(input_frames_dir, "frame_*.png")))

        if not base_frame_files and not pause_frame_files:
            print("Error: No frames found in either base or input directories.")
            return

        frame_files = sorted(base_frame_files + pause_frame_files)
        # Use absolute paths for reliable checking
        pause_frame_paths = {os.path.abspath(p) for p in pause_frame_files}
        print(f"Found {len(base_frame_files)} base frames and {len(pause_frame_files)} pause frames.")

    else:
        print(f"Using single directory '{input_frames_dir}' for all frames.")
        # Original logic: find all frame files and sort them numerically
        frame_files = sorted(glob.glob(os.path.join(input_frames_dir, "frame_*.png")))

    if not frame_files:
        print(f"Error: No frame files found to process.")
        return

    num_frames = len(frame_files)
    if num_frames == 0:
        print("No frames found to process. Exiting.")
        return

    print(f"Found a total of {num_frames} frames to process.")


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
            if base_frames_dir:
                # New logic: Check if the frame is in the pause frames set
                if os.path.abspath(frame_path) in pause_frame_paths:
                    is_pause_frame = True
            else:
                # Original logic: Check based on frame number and intervals
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
        # Use subprocess.Popen to run FFmpeg and stream its output in real-time.
        # FFmpeg prints its progress to stderr, so we redirect it to a PIPE.
        # We set stdout to DEVNULL as we don't need to capture it.
        process = subprocess.Popen(
            ffmpeg_command,
            stdout=subprocess.DEVNULL,  # Hide standard output
            stderr=subprocess.PIPE,     # Capture standard error to read progress
            text=True,                  # Decode stderr as text
            bufsize=1,                  # Line-buffered
            universal_newlines=True     # Ensure cross-platform newline handling
        )

        # Read and print each line from stderr in real-time
        print("--- FFmpeg Progress ---")
        for line in process.stderr:
            # The 'frame=' line from ffmpeg is printed with a carriage return `\r`
            # to overwrite the previous line. We use end='' to preserve this behavior.
            print(line, end='')
        print("\n--- End of FFmpeg Output ---")

        # Wait for the process to complete
        process.wait()

        # Check if the process exited with an error
        if process.returncode != 0:
            print(f"\nError: FFmpeg exited with non-zero code: {process.returncode}")
            print("The video may not have been created correctly.")
        else:
            print(f"\nVideo '{output_video_path}' created successfully!")

    except FileNotFoundError:
        print("\nError: FFmpeg not found.")
        print("Please ensure FFmpeg is installed on your system and accessible via your system's PATH.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        # --- Step 4: Clean up temporary file ---
        if os.path.exists(concat_list_path):
            os.remove(concat_list_path)
            print(f"\nCleaned up temporary file: '{concat_list_path}'")



if __name__ == "__main__":
    create_video_with_pauses(
        input_frames_dir="/home/mani/Central/HaVid/S02A08I21/GVHMR/front/preprocess/VitPose-overlay-window3",
        output_video_path="/home/mani/Central/HaVid/S02A08I21/GVHMR/front/preprocess/my_custom_video.mp4",
        base_fps=15,
        pause_duration_seconds=2,
        initial_pause_frame=286,
        subsequent_pause_interval=60
    )