import cv2
import os

VIDEO_PATH = 'VID3.mp4'
OUTPUT_DIR = 'output'
SEGMENT_DURATION = 3  # seconds

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Open the video file
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: Unable to open", VIDEO_PATH)
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    print("Error: FPS is 0.")
    exit()

# Calculate the number of frames per segment
frames_per_segment = int(fps * SEGMENT_DURATION)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
segment_index = 0

while True:
    frames = []
    # Read frames for one segment
    for _ in range(frames_per_segment):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    if not frames:
        break  # No more frames

    # Define output file for the current segment with avi extension for better compatibility
    output_file = os.path.join(OUTPUT_DIR, f"segment_{segment_index}.avi")
    writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # Write frames to the segment file while flipping vertically to correct the orientation
    for frame in frames:
        writer.write(cv2.flip(frame, 0))
    writer.release()

    print(f"Wrote segment {segment_index} ({len(frames)} frames)")
    segment_index += 1

cap.release()
print("Video segmentation complete.")