import os
# Disable MKL threading to avoid symbol conflicts
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1' 
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Remove or comment out the problematic MKL_THREADING_LAYER
# os.environ['MKL_THREADING_LAYER'] = 'GNU'

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import peakutils
import yaml
from tqdm import tqdm

def scale(img, xScale, yScale):
    res = cv2.resize(img, None, fx=xScale, fy=yScale, interpolation=cv2.INTER_AREA)
    return res


def crop(infile, height, width):
    im = Image.open(infile)
    imgwidth, imgheight = im.size
    for i in range(imgheight // height):
        for j in range(imgwidth // width):
            box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
            yield im.crop(box)


def averagePixels(path):
    r, g, b = 0, 0, 0
    count = 0
    pic = Image.open(path)
    for x in range(pic.size[0]):
        for y in range(pic.size[1]):
            imgData = pic.load()
            tempr, tempg, tempb = imgData[x, y]
            r += tempr
            g += tempg
            b += tempb
            count += 1
    return (r / count), (g / count), (b / count), count

def convert_frame_to_grayscale(frame):
    grayframe = None
    gray = None
    if frame is not None:
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = scale(gray, 1, 1)
        grayframe = scale(gray, 1, 1)
        gray = cv2.GaussianBlur(gray, (9, 9), 0.0)
    return grayframe, gray

def prepare_dirs(keyframePath, imageGridsPath, csvPath):
    if not os.path.exists(keyframePath):
        os.makedirs(keyframePath)
    if not os.path.exists(imageGridsPath):
        os.makedirs(imageGridsPath)
    if not os.path.exists(csvPath):
        os.makedirs(csvPath)


def plot_metrics(indices, lstfrm, lstdiffMag):
    y = np.array(lstdiffMag)
    plt.plot(indices, y[indices], "x")
    l = plt.plot(lstfrm, lstdiffMag, 'r-')
    plt.xlabel('frames')
    plt.ylabel('pixel difference')
    plt.title("Pixel value differences from frame to frame and the peak values")
    plt.show()


def extract_keyframes_from_images(image_filenames, n, threshold=0.3):
    """
    Extract n most important frames from a list of image filenames.
    
    Args:
        image_filenames (list): List of image file paths
        n (int): Number of keyframes to extract
        threshold (float): Threshold for peak detection (default: 0.3)
    
    Returns:
        list: List of tuples (frame_index, filename, diff_magnitude) for the n most important frames
    """
    if len(image_filenames) < 2:
        return [(0, image_filenames[0], 0)] if image_filenames else []
    
    lstdiffMag = []
    lastFrame = None
    
    # Calculate pixel differences between consecutive frames
    print(f"Processing {len(image_filenames)} frames...")
    for i, filename in enumerate(tqdm(image_filenames, desc="Analyzing frames")):
        if not os.path.exists(filename):
            print(f"Warning: File not found: {filename}")
            continue
            
        frame = cv2.imread(filename)
        if frame is None:
            print(f"Warning: Could not read frame: {filename}")
            continue
            
        grayframe, blur_gray = convert_frame_to_grayscale(frame)
        
        if i == 0:
            lastFrame = blur_gray
            lstdiffMag.append(0)
            continue
        
        diff = cv2.subtract(blur_gray, lastFrame)
        diffMag = cv2.countNonZero(diff)
        lstdiffMag.append(diffMag)
        lastFrame = blur_gray
        
        # Show progress every 100 frames
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(image_filenames)} frames, current diff: {diffMag}")
    
    if len(lstdiffMag) < 3:
        # Return first frame if not enough data for peak detection
        return [(0, image_filenames[0], lstdiffMag[0] if lstdiffMag else 0)]
    
    print("Finding peaks...")
    # Find peaks using the same logic as keyframeDetection
    y = np.array(lstdiffMag)
    base = peakutils.baseline(y, 2)
    indices = peakutils.indexes(y - base, threshold, min_dist=1)
    
    print(f"Found {len(indices)} initial peaks")
    
    # If we have more peaks than requested, select the n highest magnitude ones
    if len(indices) > n:
        ranked_indices = sorted(indices, key=lambda i: lstdiffMag[i], reverse=True)[:n]
        indices = sorted(ranked_indices)
        print(f"Selected top {n} peaks by magnitude")
    elif len(indices) < n:
        # If we don't have enough peaks, add frames with highest differences
        all_indices = list(range(len(lstdiffMag)))
        remaining_indices = [i for i in all_indices if i not in indices]
        additional_indices = sorted(remaining_indices, key=lambda i: lstdiffMag[i], reverse=True)[:n-len(indices)]
        indices = sorted(indices + additional_indices)
        print(f"Added {len(additional_indices)} additional frames to reach {n} keyframes")
    
    # Return the keyframes with their metadata
    keyframes = []
    for idx in indices:
        if idx < len(image_filenames):
            keyframes.append((idx, image_filenames[idx], lstdiffMag[idx]))
    
    return keyframes

def progressive_keyframe_extraction(test_image_dir, frame_step=30, threshold=0.3):
    """
    Progressive keyframe extraction from test folder.
    Analyzes first 30 frames for 1 peak, first 60 frames for 2 peaks, etc.
    
    Args:
        test_image_dir (str): Directory containing test frames
        frame_step (int): Step size for progressive analysis (default: 30)
        threshold (float): Threshold for peak detection (default: 0.3)
    
    Returns:
        dict: Dictionary with window sizes as keys and keyframe results as values
    """
    # Get all image files from the directory
    if not os.path.exists(test_image_dir):
        raise ValueError(f"Test image directory does not exist: {test_image_dir}")
    
    # Get sorted list of image files
    image_files = []
    print("Scanning directory for images...")
    for filename in sorted(os.listdir(test_image_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_files.append(os.path.join(test_image_dir, filename))
    
    if not image_files:
        raise ValueError(f"No image files found in directory: {test_image_dir}")
    
    print(f"Found {len(image_files)} image files")
    
    results = {}
    current_window = frame_step
    peak_count = 1
    
    # Calculate total windows for progress bar
    total_windows = (len(image_files) // frame_step) + (1 if len(image_files) % frame_step != 0 else 0)
    
    # Progressive analysis
    with tqdm(total=total_windows, desc="Progressive analysis") as pbar:
        while current_window <= len(image_files):
            window_files = image_files[:current_window]
            print(f"\nAnalyzing window {current_window} (first {current_window} frames) for {peak_count} peaks...")
            
            keyframes = extract_keyframes_from_images(window_files, peak_count, threshold)
            
            results[current_window] = {
                'peaks_requested': peak_count,
                'keyframes_found': len(keyframes),
                'keyframes': keyframes
            }
            
            # Show intermediate result
            print(f"Window {current_window}: Found {len(keyframes)} keyframes")
            for i, (frame_idx, filename, diff_mag) in enumerate(keyframes):
                print(f"  Keyframe {i+1}: Frame {frame_idx}, Diff: {diff_mag:.0f}")
            
            # Move to next window
            current_window += frame_step
            peak_count += 1
            pbar.update(1)
        
        # Handle remaining frames if any
        if len(image_files) % frame_step != 0:
            final_window = len(image_files)
            final_peaks = (final_window // frame_step) + 1
            print(f"\nFinal window {final_window} (all {final_window} frames) for {final_peaks} peaks...")
            
            keyframes = extract_keyframes_from_images(image_files, final_peaks, threshold)
            
            results[final_window] = {
                'peaks_requested': final_peaks,
                'keyframes_found': len(keyframes),
                'keyframes': keyframes
            }
            
            # Show final result
            print(f"Final window {final_window}: Found {len(keyframes)} keyframes")
            for i, (frame_idx, filename, diff_mag) in enumerate(keyframes):
                print(f"  Keyframe {i+1}: Frame {frame_idx}, Diff: {diff_mag:.0f}")
            
            pbar.update(1)
    
    return results

def run_progressive_keyframe_test(config_path="/home/mani/Repos/hcdt/config/exp/Stack_v2.yaml"):
    """
    Run progressive keyframe extraction using test directory from config file.
    
    Args:
        config_path (str): Path to the config YAML file
    
    Returns:
        dict: Results from progressive keyframe extraction
    """
    # Load config
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    test_image_dir = config.get('test_image_dir')
    frame_step = config.get('test_frame_step', 30)
    
    if not test_image_dir:
        raise ValueError("test_image_dir not found in config file")
    
    print(f"Running progressive keyframe extraction on: {test_image_dir}")
    print(f"Using frame step: {frame_step}")
    
    results = progressive_keyframe_extraction(test_image_dir, frame_step)
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY - Progressive Keyframe Extraction Results:")
    print("="*60)
    for window_size, result in results.items():
        print(f"Window {window_size}: {result['keyframes_found']} keyframes found (requested {result['peaks_requested']})")
    
    return results

if __name__ == "__main__":
    # Run progressive keyframe extraction test
    results = run_progressive_keyframe_test()