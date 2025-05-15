from PIL import Image, ImageDraw

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
    draw = ImageDraw.Draw(img)
    left_up = (x - point_radius, y - point_radius)
    right_down = (x + point_radius, y + point_radius)
    draw.ellipse([left_up, right_down], fill=point_color, outline=point_color)
    if save_path:
        img.save(save_path)
    else:
        return img
