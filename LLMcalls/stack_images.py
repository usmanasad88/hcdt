from PIL import Image
import os
#camera_frames= 
prompt="You will be given two images. Predict the target of the gaze of the person in the first image. The second image has a heatmap of the gaze target. Predict the target of the gaze of the first image again based on the heatmap. If the gaze target is not clear, output unclear. If the person is looking in the camera, output camera. If the person is looking at a person, output person. If the person is looking at an object, output object. If the person is looking at a place, output place. If the person is looking at a screen, output screen. If the person is looking at a book, output book. If the person is looking at a table, output table. If the person is looking at a wall, output wall. If the person is looking at a door, output door. If the person is looking at a window, output window. If the person is looking at a floor, output floor. If the person is looking at the camera, output camera."
results_store_file="LLMcalls/gazetargets.txt"
output_combine_path="/home/mani/Central/Cooking1/Stack/cam1/output_combined_frames/"

# Combine the image and the heatmap
for second in range(0, 42):
    for fractionsecond in range(0, 5):
        timestamp = f"{second:01d}.{2*fractionsecond:01d}"
        image_path = f"/home/mani/Central/Cooking1/Stack/cam1/output_frames/cam1-frame-{timestamp}.jpg"
        heatmap_path = f"/home/mani/Central/Cooking1/Stack/cam1/output_heatmaps/heatmap_cam1-frame-{timestamp}.jpg"

        # Open heatmap to get its resolution
        try:
            heatmap = Image.open(heatmap_path)
            w_hm, h_hm = heatmap.size
        except Exception as e:
            print(f"Could not open heatmap {heatmap_path}: {e}")
            continue

        # Open and resize the image to heatmap resolution
        try:
            img = Image.open(image_path)
            img_resized = img.resize((w_hm, h_hm), Image.LANCZOS)
        except Exception as e:
            print(f"Could not open or resize image {image_path}: {e}")
            continue

        # Combine: side-by-side (image left, heatmap right)
        combined = Image.new("RGB", (w_hm, h_hm * 2))
        combined.paste(img_resized, (0, 0))
        combined.paste(heatmap, (0, h_hm))

        combined_filename = f"combined-frame-{timestamp}.jpg"
        combined_path = os.path.join(output_combine_path, combined_filename)
        combined.save(combined_path)
        print(f"Saved combined image: {combined_path}")