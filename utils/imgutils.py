from PIL import Image, ImageDraw
import os
import subprocess

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
    
if __name__ == "__main__":
    input_folders = [
        "/home/mani/Central/Cooking1/aria_frames",
        "/home/mani/Central/Cooking1/cam1_frames"
    ]
    output_folder = "/home/mani/Central/Cooking1/combined_frames"
    stack_images_from_folders(
        input_folders=input_folders,
        output_folder=output_folder,
        direction="horizontal",  # or "horizontal" if you want side-by-side
        resize_mode="smaller",  # or "larger" if you want to use the largest dimensions
        background_color=(0, 0, 0)
    )
    print(f"Combined frames saved to {output_folder}")