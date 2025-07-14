import subprocess
import matplotlib.pyplot as plt
import json
import re
from PIL import Image, ImageDraw
import os
import subprocess
import base64
import tempfile

def encode_file_to_base64(file_path):
    """Encodes a file to a base64 string."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None
    try:
        with open(file_path, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode("utf-8")
        return encoded_string
    except Exception as e:
        print(f"Error encoding file {file_path}: {e}")
        return None
    
def extract_frames_ffmpeg(video_path, output_folder, interval_sec=1.0, prefix="frame"):
    os.makedirs(output_folder, exist_ok=True)
    fps = 1.0 / interval_sec
    output_pattern = os.path.join(output_folder, f"{prefix}-%05d.jpg")
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}",
        output_pattern
    ]
    subprocess.run(cmd, check=True)


def stack_images_from_folders(
    input_folders,
    output_folder,
    direction="vertical",
    resize_mode="smaller",
    background_color=(0, 0, 0)
):
    """
    Stack images from multiple input folders (by filename order) and save to output_folder.
    Keeps the aspect ratio of the original images by padding with background_color.
    Args:
        input_folders (list of str): List of folders containing images to stack.
        output_folder (str): Folder to save the stacked images.
        direction (str): 'vertical' or 'horizontal'.
        resize_mode (str): 'smaller' or 'larger' - resize all images to the smallest or largest dimensions found.
        background_color (tuple): RGB color for background if images are different sizes.
    """
    os.makedirs(output_folder, exist_ok=True)
    # Get sorted list of filenames present in all folders
    filenames = sorted(
        set.intersection(*[set(os.listdir(folder)) for folder in input_folders])
    )
    for fname in filenames:
        images = []
        sizes = []
        for folder in input_folders:
            img_path = os.path.join(folder, fname)
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            sizes.append(img.size)
        # Determine resize target
        if resize_mode == "smaller":
            target_size = (min(w for w, h in sizes), min(h for w, h in sizes))
        elif resize_mode == "larger":
            target_size = (max(w for w, h in sizes), max(h for w, h in sizes))
        else:
            raise ValueError("resize_mode must be 'smaller' or 'larger'")

        # Resize images to fit within target_size, keeping aspect ratio, and pad
        padded_images = []
        for img in images:
            img_ratio = img.width / img.height
            target_ratio = target_size[0] / target_size[1]
            if img_ratio > target_ratio:
                # Fit to width
                new_w = target_size[0]
                new_h = int(new_w / img_ratio)
            else:
                # Fit to height
                new_h = target_size[1]
                new_w = int(new_h * img_ratio)
            img_resized = img.resize((new_w, new_h), Image.LANCZOS)
            # Create background and paste centered
            padded = Image.new("RGB", target_size, background_color)
            offset = ((target_size[0] - new_w) // 2, (target_size[1] - new_h) // 2)
            padded.paste(img_resized, offset)
            padded_images.append(padded)

        # Stack
        if direction == "vertical":
            total_width = target_size[0]
            total_height = target_size[1] * len(padded_images)
            stacked_img = Image.new("RGB", (total_width, total_height), background_color)
            for idx, img in enumerate(padded_images):
                stacked_img.paste(img, (0, idx * target_size[1]))
        elif direction == "horizontal":
            total_width = target_size[0] * len(padded_images)
            total_height = target_size[1]
            stacked_img = Image.new("RGB", (total_width, total_height), background_color)
            for idx, img in enumerate(padded_images):
                stacked_img.paste(img, (idx * target_size[0], 0))
        else:
            raise ValueError("direction must be 'vertical' or 'horizontal'")
        # Save
        out_path = os.path.join(output_folder, fname)
        stacked_img.save(out_path)

def draw_point_on_image(image_path, y, x, save_path=None, point_color='red', point_radius=8):
    """
    Draw a point at (y, x) on the image and save or return the result.

    Args:
        image_path (str): Path to the image file.
        y (int or float): Y coordinate (vertical).
        x (int or float): X coordinate (horizontal).
        save_path (str, optional): If provided, saves the image to this path.
        point_color (str): Color of the point.
        point_radius (int): Radius of the point.

    Returns:
        PIL.Image.Image: The image with the point drawn (if save_path is None).
    """
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    # Convert normalized coordinates to pixel locations
    x_px = int((x / 1000.0) * width)
    y_px = int((y / 1000.0) * height)
    draw = ImageDraw.Draw(img)
    left_up = (x_px - point_radius, y_px - point_radius)
    right_down = (x_px + point_radius, y_px + point_radius)
    draw.ellipse([left_up, right_down], fill=point_color, outline=point_color)
    if save_path:
        img.save(save_path)
    else:
        return img
    
def draw_multiple_points_on_image(image_path, points, save_path=None, point_color='red', point_radius=8):
    """
    Draw multiple points on an image at the given [y, x] coordinates and save or return the result.

    Args:
        image_path (str): Path to the image file.
        points (list of tuples/lists): A list of [y, x] coordinates.
                                       Each y and x should be scaled from 0 to 1000.
        save_path (str, optional): If provided, saves the image to this path.
                                   Otherwise, the image object is returned.
        point_color (str): Color of the points.
        point_radius (int): Radius of the points.

    Returns:
        PIL.Image.Image: The image with the points drawn (if save_path is None).
                         Returns None if save_path is provided.
    """
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    draw = ImageDraw.Draw(img)

    for point in points:
        y, x = point
        # Convert normalized coordinates to pixel locations
        x_px = int((x / 1000.0) * width)
        y_px = int((y / 1000.0) * height)

        # Define the bounding box for the ellipse (dot)
        left_up = (x_px - point_radius, y_px - point_radius)
        right_down = (x_px + point_radius, y_px + point_radius)
        
        draw.ellipse([left_up, right_down], fill=point_color, outline=point_color)

    if save_path:
        # Ensure the directory for save_path exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img.save(save_path)
        print(f"Image with points saved to {save_path}")
        return None
    else:
        return img


def resize_image_to_largest_dimension(image_path, target_largest_dimension, save_path):
    """
    Opens an image, resizes it so its largest dimension equals target_largest_dimension
    while maintaining aspect ratio, and saves it.

    Args:
        image_path (str): Path to the input image file.
        target_largest_dimension (int): The desired size for the largest dimension of the image.
        save_path (str): Path to save the resized image.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        original_width, original_height = img.size

        if original_width == 0 or original_height == 0:
            print(f"Warning: Image at {image_path} has zero dimension.")
            return

        aspect_ratio = original_width / original_height

        if original_width >= original_height:
            # Width is the largest or image is square
            new_width = target_largest_dimension
            new_height = int(new_width / aspect_ratio)
        else:
            # Height is the largest
            new_height = target_largest_dimension
            new_width = int(new_height * aspect_ratio)
        
        # Ensure new dimensions are at least 1 pixel
        new_width = max(1, new_width)
        new_height = max(1, new_height)

        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Ensure the directory for save_path exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        resized_img.save(save_path)
        print(f"Resized image saved to {save_path}")

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def overlay_frame_numbers_on_folder(input_folder, output_folder=None, font_size=36, font_color='white', position='top-left'):
    """
    Overlay frame numbers on all image files in an input folder.

    Args:
        input_folder (str): Path to the folder containing input images.
        output_folder (str, optional): Path to save images with frame numbers. 
                                     If None, saves to input_folder + '_numbered'.
        font_size (int): Size of the font for frame numbers.
        font_color (str): Color of the frame number text.
        position (str): Position of the frame number ('top-left', 'top-right', 'bottom-left', 'bottom-right').

    Returns:
        str: Path to the output folder.
    """
    from PIL import ImageFont
    
    # Set output folder if not provided
    if output_folder is None:
        output_folder = f"{input_folder}_numbered"
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files and sort them
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [f for f in os.listdir(input_folder) 
                   if os.path.splitext(f.lower())[1] in image_extensions]
    image_files.sort()
    
    if not image_files:
        print(f"No image files found in {input_folder}")
        return output_folder
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()
            print("Warning: Using default font as system fonts not found")
    
    for frame_num, filename in enumerate(image_files, 1):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        try:
            # Open image
            img = Image.open(input_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            
            # Prepare frame number text
            frame_text = f"Frame {frame_num:04d}"
            
            # Get text dimensions
            bbox = draw.textbbox((0, 0), frame_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Calculate position based on user preference
            img_width, img_height = img.size
            margin = 20
            
            if position == 'top-left':
                text_x, text_y = margin, margin
            elif position == 'top-right':
                text_x, text_y = img_width - text_width - margin, margin
            elif position == 'bottom-left':
                text_x, text_y = margin, img_height - text_height - margin
            elif position == 'bottom-right':
                text_x, text_y = img_width - text_width - margin, img_height - text_height - margin
            else:
                # Default to top-left
                text_x, text_y = margin, margin
            
            # Draw text with outline for better visibility
            outline_width = 2
            # Draw outline
            for dx in range(-outline_width, outline_width + 1):
                for dy in range(-outline_width, outline_width + 1):
                    if dx != 0 or dy != 0:
                        draw.text((text_x + dx, text_y + dy), frame_text, font=font, fill='black')
            
            # Draw main text
            draw.text((text_x, text_y), frame_text, font=font, fill=font_color)
            
            # Save image
            img.save(output_path)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    print(f"Processed {len(image_files)} images. Output saved to {output_folder}")
    return output_folder

def image_to_base64(image_path, target_largest_dimension=None):
    """
    Optionally resizes an image to a target largest dimension and encodes it to base64.
    If target_largest_dimension is None, the original image is encoded without resizing.

    Args:
        image_path (str): Path to the input image file.
        target_largest_dimension (int, optional): The desired size for the largest dimension of the image.
                                                 If None, no resizing is performed. Defaults to None.

    Returns:
        str: Base64 encoded string of the (potentially resized) image, or None if an error occurs.
    """
    try:
        if target_largest_dimension is not None:
            # Create a temporary file for the resized image
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # Resize the image to the target dimension
                resize_image_to_largest_dimension(image_path, target_largest_dimension, temp_path)
                # Encode the resized image to base64
                base64_string = encode_file_to_base64(temp_path)
            finally:
                # Clean up the temporary file
                os.unlink(temp_path)
        else:
            # No resizing, encode the original image directly
            base64_string = encode_file_to_base64(image_path)
            
        return base64_string
        
    except Exception as e:
        print(f"Error processing image {image_path} for base64 encoding: {e}")
        # Clean up temp file if it was created and an error occurred before unlinking
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        return None
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        # Clean up temp file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        return None
    
def resize_image_to_model_size(image_path, w_model, h_model, save_path=None):
    """
    Resize the input image to the model's expected size.
    """
    img = Image.open(image_path)
    img_resized = img.resize((w_model, h_model), Image.LANCZOS)
    if save_path:
        img_resized.save(save_path)
        return save_path
    return img_resized

def plot_points_on_image(image_path, points, point_color='red', point_size=50):
    """
    Plots given (x, y) points on the image.
    """
    img = Image.open(image_path)
    plt.imshow(img)
    xs, ys = zip(*points)
    plt.scatter(xs, ys, c=point_color, s=point_size)
    plt.axis('off')
    plt.show()

def extract_xy_coords(data):
    """
    Recursively extract all dicts with 'x' and 'y' keys from data.
    """
    coords = []
    if isinstance(data, dict):
        if "x" in data and "y" in data:
            coords.append(data)
        for v in data.values():
            coords.extend(extract_xy_coords(v))
    elif isinstance(data, list):
        for item in data:
            coords.extend(extract_xy_coords(item))
    return coords


def plot_points_phase2(phase2_file, frames_folder, output_folder): # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        with open(phase2_file, 'r') as f:
            phase2_data = json.load(f)
        
        # Get all frame files and sort them
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        frame_files = [f for f in os.listdir(frames_folder) 
                      if os.path.splitext(f.lower())[1] in image_extensions]
        frame_files.sort()
        
        if not frame_files:
            print(f"No image files found in {frames_folder}")
        else:
            print(f"Found {len(frame_files)} frame files")
            
        # Create a dictionary for quick lookup of predictions by frame number
        predictions_dict = {entry["prediction_frame"]: entry["predicted_hand_positions"] for entry in phase2_data}
        
        print(f"Loaded predictions for frames: {sorted(predictions_dict.keys())}")
        
        # Process each frame file
        for i, frame_file in enumerate(frame_files, 1):
            frame_number = i  # Frame numbering starts from 1
            frame_path = os.path.join(frames_folder, frame_file)
            output_path = os.path.join(output_folder, f"frame_{frame_number:04d}_with_predictions.png")
            
            # Check if we have predictions for this frame
            if frame_number in predictions_dict:
                pred_positions = predictions_dict[frame_number]
                
                # Extract hand positions (note: coordinates are already in 0-1000 range)
                left_hand_point = [pred_positions["left_hand_y"], pred_positions["left_hand_x"]]
                right_hand_point = [pred_positions["right_hand_y"], pred_positions["right_hand_x"]]
                
                # Create points list with different colors
                points = [left_hand_point, right_hand_point]
                
                print(f"Frame {frame_number}: Adding predicted hand positions")
                print(f"  Left hand: ({pred_positions['left_hand_x']:.1f}, {pred_positions['left_hand_y']:.1f})")
                print(f"  Right hand: ({pred_positions['right_hand_x']:.1f}, {pred_positions['right_hand_y']:.1f})")
                
                # Load image and draw points
                img = Image.open(frame_path).convert("RGB")
                width, height = img.size
                draw = ImageDraw.Draw(img)
                
                # Draw left hand (red circle)
                left_x_px = int((pred_positions["left_hand_x"] / 1000.0) * width)
                left_y_px = int((pred_positions["left_hand_y"] / 1000.0) * height)
                point_radius = 12
                
                # Left hand - Red
                draw.ellipse([
                    (left_x_px - point_radius, left_y_px - point_radius),
                    (left_x_px + point_radius, left_y_px + point_radius)
                ], fill='red', outline='darkred', width=2)
                
                # Right hand - Blue
                right_x_px = int((pred_positions["right_hand_x"] / 1000.0) * width)
                right_y_px = int((pred_positions["right_hand_y"] / 1000.0) * height)
                
                draw.ellipse([
                    (right_x_px - point_radius, right_y_px - point_radius),
                    (right_x_px + point_radius, right_y_px + point_radius)
                ], fill='blue', outline='darkblue', width=2)
                
                # Add text labels
                from PIL import ImageFont
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
                except (OSError, IOError):
                    font = ImageFont.load_default()
                
                # Left hand label
                draw.text((left_x_px + 15, left_y_px - 10), "L", fill='red', font=font)
                # Right hand label  
                draw.text((right_x_px + 15, right_y_px - 10), "R", fill='blue', font=font)
                
                # Add frame info
                frame_info = f"Frame {frame_number} - Predicted Hand Positions"
                draw.text((10, 10), frame_info, fill='white', font=font)
                
                # Save the image
                img.save(output_path)
                
            else:
                # No predictions for this frame, just copy the original
                img = Image.open(frame_path).convert("RGB")
                draw = ImageDraw.Draw(img)
                
                # Add frame info indicating no predictions
                from PIL import ImageFont
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
                except (OSError, IOError):
                    font = ImageFont.load_default()
                
                frame_info = f"Frame {frame_number} - No Predictions"
                draw.text((10, 10), frame_info, fill='yellow', font=font)
                img.save(output_path)
                
                print(f"Frame {frame_number}: No predictions available")
        
        print(f"\n✅ Processed {len(frame_files)} frames")
        print(f"📁 Output saved to: {output_folder}")
        print(f"🔴 Red circles = Left hand predictions")
        print(f"🔵 Blue circles = Right hand predictions")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find {phase2_file}")
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # input_folder = "/home/mani/Central/Cooking1/combined_frames"
    # output_folder = "/home/mani/Central/Cooking1/combined_frames_numbered"
    # overlay_frame_numbers_on_folder(input_folder, font_size=48, position='top-right')

    # image_path = "/home/mani/Central/HaVid/S01A02I01S1/frame_0001.png"  
    # points=[[859, 696]]
    # base_path, _ = os.path.splitext(image_path)
    # new_save_path = f"{base_path}_copy.png"
    # draw_multiple_points_on_image(image_path,points,new_save_path)
    # resize_image_to_largest_dimension(image_path , 640, image_path)

    # input_folders = [
    #     "/home/mani/Central/Cooking1/aria_frames",
    #     "/home/mani/Central/Cooking1/cam1_frames"
    # ]
    # output_folder = "/home/mani/Central/Cooking1/combined_frames"
    # stack_images_from_folders(
    #     input_folders=input_folders,
    #     output_folder=output_folder,
    #     direction="horizontal",  # or "horizontal" if you want side-by-side
    #     resize_mode="smaller",  # or "larger" if you want to use the largest dimensions
    #     background_color=(0, 0, 0)
    # )
    # print(f"Combined frames saved to {output_folder}")

if __name__ == "__main__":
    phase2_file = "data/HAViD/phase2_icl_result_window_3.json"
    frames_folder = "/home/mani/Central/HaVid/S02A08I21/GVHMR/front/preprocess/vitpose_temp"
    output_folder = "/home/mani/Central/HaVid/S02A08I21/GVHMR/front/preprocess/VitPose-overlay-window3"
    plot_points_phase2(phase2_file, frames_folder, output_folder)
    