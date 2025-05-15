from PIL import Image, ImageDraw
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rungemma import run_llama_mtmd
from utils.imgutils import draw_point_on_image  
from utils.textutils import parse_yx



#camera_frames= 
prompt="""Using the provided image and gaze heatmap, determine the most likely object of the person's gaze. Output your prediction as a single word. Possible outputs include object names, "camera", or "unclear"."""
prompt2="""Point to the gaze target location of the person. The answer should follow the json format: [{"point": <point>, "label": <label1>}, ...]. The points are in [y, x] format normalized to 0-1000."""
prompt3="""Point to the head of the person in the image.  The answer should follow the json format: [{"point": <point>, "label": <label1>}, ...]. The points are in [y, x] format normalized to 0-1000."""
responsetypePixels=False

results_store_file="LLMcalls/gazetargets.txt"
output_combine_path="/home/mani/Central/Cooking1/Stack/cam1/output_combined_frames/"
heatmap_path = f"/home/mani/Central/Cooking1/Stack/cam1/output_heatmaps/heatmap_cam1-frame-"
      
with open(results_store_file, "w") as f_out:
    for second in range(0, 42):
        for fractionsecond in range(0, 5):
            timestamp = f"{second:01d}.{2*fractionsecond:01d}"
            image_path = f"/home/mani/Central/Cooking1/Stack/cam1/output_combined_frames/combined-frame-{timestamp}.jpg"
            #image_path = "/home/mani/Central/Cooking1/Stack/cam1/output_frames/cam1-frame-6.8.jpg"
            response = run_llama_mtmd(prompt, image_path)

            if responsetypePixels:                
                y, x = parse_yx(response)
                draw_point_on_image(image_path, y, x, save_path=image_path.replace(".png", "_point.jpg"), point_color='red', point_radius=8)

                print(response)
            
            print(f"{timestamp}\t{response}\n")
            # Store the raw responses
            f_out.write(f"{timestamp}\t{response}\n")

