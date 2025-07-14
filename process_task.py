#!/usr/bin/env python3
"""
Process Task Pipeline for Phase 2 Experiments
==============================================

This script handles the complete preprocessing pipeline for a new experiment:
1. Creates necessary directories
2. Extracts frames from videos using ffmpeg
3. Runs GVHMR pose estimation
4. Converts GVHMR output to HumanML3D format
5. Runs Gazelle gaze estimation

The script uses three different conda environments:
- gvhmr: for pose estimation
- smplestx: for pose format conversion
- gazelle: for gaze estimation
"""

import os
import subprocess
import sys
import argparse
from pathlib import Path
import shutil
import logging
import yaml
import json

# Import utility functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from utils.imgutils import overlay_frame_numbers_on_folder
from utils.motionutils import get_hand_xy_positions

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Experiment configuration
# exp_name = "Stack_v2"

# test_input_folder = "/home/mani/Central/Stack/exp2"
# test_video_names = ["cam01.mp4"]
# test_egocentric_video_names = []

# example_input_folder = "/home/mani/Central/Stack/cam1"
# example_video_names = ["cam01.mp4"]
# example_egocentric_video_names = []

exp_name = "cooking"

test_input_folder = "/home/mani/Central/Cooking1/FairCooking/fair_cooking_05_2"
test_video_names = ["cam03.mp4", "cam01.mp4", "cam02.mp4", "cam04.mp4"]
test_egocentric_video_names = ["aria02_214-1.mp4"]

example_input_folder = "/home/mani/Central/Cooking1/FairCooking/fair_cooking_05_4"
example_video_names = ["cam03.mp4", "cam01.mp4", "cam02.mp4", "cam04.mp4"]
example_egocentric_video_names = ["aria02_214-1.mp4"]

config_path = "/home/mani/Repos/hcdt/config/exp/"

# Conda environments
GVHMR_ENV = "gvhmr"
SMPLESTX_ENV = "smplestx"
GAZELLE_ENV = "gazelle"

# External tool paths
GVHMR_DEMO_PATH = "/home/mani/GVHMR/tools/demo/demo.py"
GVHMR_TO_HML3D_PATH = "/home/mani/SMPLest-X/main/gvhmr_to_hml3d.py"
GAZELLE_SCRIPT_PATH = "/home/mani/gazelle/scripts/test2.py"

# Global flag for dry run mode
DRY_RUN = False


def run_command(command, env_name=None, cwd=None):
    """Run a command with optional conda environment activation."""
    if env_name:
        # Activate conda environment and run command
        full_command = f"conda run -n {env_name} {command}"
    else:
        full_command = command
    
    logger.info(f"Running: {full_command}")
    if cwd:
        logger.info(f"Working directory: {cwd}")
    
    if DRY_RUN:
        logger.info("DRY RUN: Command would be executed but skipping actual execution")
        return None
    
    try:
        result = subprocess.run(
            full_command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"Command completed successfully")
        if result.stdout.strip():
            logger.info(f"STDOUT: {result.stdout.strip()}")
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with return code {e.returncode}")
        if e.stderr.strip():
            logger.error(f"STDERR: {e.stderr.strip()}")
        if e.stdout.strip():
            logger.error(f"STDOUT: {e.stdout.strip()}")
        raise


def create_directories():
    """Create necessary directory structure."""
    logger.info("Creating directory structure...")
    
    dirs_to_create = [
        # Test directories
        f"{test_input_folder}/frames",
        f"{test_input_folder}/GVHMR",
        f"{test_input_folder}/gazelle_output",
        
        # Example directories
        f"{example_input_folder}/frames",
        f"{example_input_folder}/GVHMR",
        f"{example_input_folder}/gazelle_output",
        
        # Data directories
        "/home/mani/Repos/hcdt/data/Stack",
        "/home/mani/Repos/hcdt/data/HAViD", 
        "/home/mani/Repos/hcdt/data/humanml3d"
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")


def generate_dirs_yaml():
    """Generate a YAML file with all the correct directory names."""
    logger.info("Generating directory configuration YAML...")
    
    # Determine video stem names (without extension)
    test_video_stem = Path(test_video_names[0]).stem if test_video_names else "cam01"
    example_video_stem = Path(example_video_names[0]).stem if example_video_names else "cam01"
    
    # Get folder names for naming convention
    test_folder_name = Path(test_input_folder).name
    example_folder_name = Path(example_input_folder).name
    
    dirs_config = {
        "experiment_name": exp_name,
        "test_folders": {
            "input_folder": test_input_folder,
            "frames_dir": f"{test_input_folder}/frames",
            "gvhmr_output_root": f"{test_input_folder}/GVHMR",
            "gvhmr_video_dir": f"{test_input_folder}/GVHMR/{test_video_stem}",
            "vitpose_file": f"{test_input_folder}/GVHMR/{test_video_stem}/preprocess/vitpose.pt",
            "vitpose_frames_dir": f"{test_input_folder}/GVHMR/{test_video_stem}/preprocess/VitPose",
            "vitpose_video_overlay": f"{test_input_folder}/GVHMR/{test_video_stem}/preprocess/vitpose_video_overlay.mp4",
            "vitpose_overlay_frames_dir": f"{test_input_folder}/GVHMR/{test_video_stem}/preprocess/vitpose_overlay_frames",
            "hmr4d_results": f"{test_input_folder}/GVHMR/{test_video_stem}/hmr4d_results.pt",
            "gazelle_output_dir": f"{test_input_folder}/gazelle_output",
            "humanml3d_file": f"/home/mani/Repos/hcdt/data/humanml3d/test_{test_video_stem}.pt",
            "phase2_ground_truth_file": f"/home/mani/Repos/hcdt/data/HAViD/phase2_test_{test_video_stem}.json"
        },
        "example_folders": {
            "input_folder": example_input_folder,
            "frames_dir": f"{example_input_folder}/frames",
            "gvhmr_output_root": f"{example_input_folder}/GVHMR",
            "gvhmr_video_dir": f"{example_input_folder}/GVHMR/{example_video_stem}",
            "vitpose_file": f"{example_input_folder}/GVHMR/{example_video_stem}/preprocess/vitpose.pt",
            "vitpose_frames_dir": f"{example_input_folder}/GVHMR/{example_video_stem}/preprocess/VitPose",
            "vitpose_video_overlay": f"{example_input_folder}/GVHMR/{example_video_stem}/preprocess/vitpose_video_overlay.mp4",
            "vitpose_overlay_frames_dir": f"{example_input_folder}/GVHMR/{example_video_stem}/preprocess/vitpose_overlay_frames",
            "hmr4d_results": f"{example_input_folder}/GVHMR/{example_video_stem}/hmr4d_results.pt",
            "gazelle_output_dir": f"{example_input_folder}/gazelle_output",
            "humanml3d_file": f"/home/mani/Repos/hcdt/data/humanml3d/example_{example_video_stem}.pt",
            "phase2_ground_truth_file": f"/home/mani/Repos/hcdt/data/HAViD/phase2_example_{example_video_stem}.json"
        },
        "data_folders": {
            "stack_data_dir": "/home/mani/Repos/hcdt/data/Stack",
            "havid_data_dir": "/home/mani/Repos/hcdt/data/HAViD",
            "humanml3d_data_dir": "/home/mani/Repos/hcdt/data/humanml3d"
        },
        "config_paths": {
            "main_config": f"{config_path}{exp_name}.yaml",
            "dirs_config": f"{config_path}{exp_name}_dirs.yaml"
        },
        "phase2_files": {
            "test_vitpose_frames": f"{test_input_folder}/GVHMR/{test_video_stem}/preprocess/VitPose",
            "example_vitpose_frames": f"{example_input_folder}/GVHMR/{example_video_stem}/preprocess/VitPose",
            "test_vitpose_overlay_frames": f"{test_input_folder}/GVHMR/{test_video_stem}/preprocess/vitpose_overlay_frames",
            "example_vitpose_overlay_frames": f"{example_input_folder}/GVHMR/{example_video_stem}/preprocess/vitpose_overlay_frames",
            "test_image_dir": f"{test_input_folder}/frames",
            "example_image_dir": f"{example_input_folder}/frames",
            "test_phase2_ground_truth_file": f"/home/mani/Repos/hcdt/data/HAViD/phase2_test_{test_video_stem}.json",
            "example_phase2_ground_truth_file": f"/home/mani/Repos/hcdt/data/HAViD/phase2_example_{example_video_stem}.json"
        }
    }
    
    # Write the YAML file
    dirs_yaml_path = f"{config_path}{exp_name}_dirs.yaml"
    
    if not DRY_RUN:
        with open(dirs_yaml_path, 'w') as f:
            yaml.dump(dirs_config, f, default_flow_style=False, indent=2, sort_keys=False)
        logger.info(f"Directory configuration saved to: {dirs_yaml_path}")
    else:
        logger.info(f"DRY RUN: Would create directory configuration at: {dirs_yaml_path}")
    
    return dirs_config


def extract_frames_ffmpeg(video_path, output_dir):
    """Extract all frames from video using ffmpeg and add frame numbers."""
    logger.info(f"Extracting frames from {video_path} to {output_dir}")
    
    # Check if frames already exist
    if os.path.exists(output_dir) and not DRY_RUN:
        existing_frames = len([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
        if existing_frames > 0:
            logger.info(f"Frames already exist in {output_dir} ({existing_frames} frames). Skipping extraction.")
            return
    
    # Create output directory
    if not DRY_RUN:
        os.makedirs(output_dir, exist_ok=True)
    else:
        logger.info(f"DRY RUN: Would create directory {output_dir}")
        return
    
    # ffmpeg command to extract all frames as JPG
    command = f'ffmpeg -i "{video_path}" "{output_dir}/frame_%06d.jpg" -y'
    
    try:
        run_command(command)
        
        # Add frame numbers to the extracted frames
        if not DRY_RUN:
            logger.info(f"Adding frame numbers to extracted frames...")
            overlay_frame_numbers_on_folder(
                input_folder=output_dir,
                output_folder=output_dir,
                font_size=36,
                font_color='white',
                position='top-right'
            )
            logger.info(f"Successfully added frame numbers to all frames")
        
        logger.info(f"Successfully extracted frames to {output_dir}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to extract frames: {e}")
        raise


def extract_vitpose_overlay_frames(gvhmr_output_root, video_stem):
    """Extract frames from VitPose video overlay created by GVHMR and add frame numbers."""
    vitpose_video_path = os.path.join(gvhmr_output_root, video_stem, "preprocess", "vitpose_video_overlay.mp4")
    vitpose_frames_output_dir = os.path.join(gvhmr_output_root, video_stem, "preprocess", "vitpose_overlay_frames")
    
    # Check if VitPose video overlay exists
    if not os.path.exists(vitpose_video_path) and not DRY_RUN:
        logger.warning(f"VitPose video overlay not found at {vitpose_video_path}. Skipping frame extraction.")
        return
    
    logger.info(f"Extracting frames from VitPose video overlay: {vitpose_video_path}")
    
    # Check if frames already exist
    if os.path.exists(vitpose_frames_output_dir) and not DRY_RUN:
        existing_frames = len([f for f in os.listdir(vitpose_frames_output_dir) if f.endswith('.jpg')])
        if existing_frames > 0:
            logger.info(f"VitPose overlay frames already exist in {vitpose_frames_output_dir} ({existing_frames} frames). Skipping extraction.")
            return
    
    # Create output directory
    if not DRY_RUN:
        os.makedirs(vitpose_frames_output_dir, exist_ok=True)
    else:
        logger.info(f"DRY RUN: Would create directory {vitpose_frames_output_dir}")
        return
    
    # ffmpeg command to extract all frames from VitPose overlay video as JPG
    command = f'ffmpeg -i "{vitpose_video_path}" "{vitpose_frames_output_dir}/vitpose_frame_%06d.jpg" -y'
    
    try:
        run_command(command)
        
        # Add frame numbers to the extracted VitPose overlay frames
        if not DRY_RUN:
            frame_count = len([f for f in os.listdir(vitpose_frames_output_dir) if f.endswith('.jpg')])
            logger.info(f"Adding frame numbers to {frame_count} VitPose overlay frames...")
            overlay_frame_numbers_on_folder(
                input_folder=vitpose_frames_output_dir,
                output_folder=vitpose_frames_output_dir,
                font_size=24,
                font_color='yellow',
                position='top-left'
            )
            logger.info(f"Successfully added frame numbers to VitPose overlay frames")
        
        frame_count = len([f for f in os.listdir(vitpose_frames_output_dir) if f.endswith('.jpg')])
        logger.info(f"Successfully extracted {frame_count} frames from VitPose overlay to {vitpose_frames_output_dir}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to extract VitPose overlay frames: {e}")
        raise


def run_gvhmr_pose_estimation(video_path, output_root):
    """Run GVHMR pose estimation."""
    logger.info(f"Running GVHMR pose estimation for {video_path}")
    
    # Check if GVHMR results already exist
    video_stem = Path(video_path).stem
    hmr4d_results_path = os.path.join(output_root, video_stem, "hmr4d_results.pt")
    vitpose_frames_dir = os.path.join(output_root, video_stem, "preprocess", "VitPose")
    
    if os.path.exists(hmr4d_results_path) and not DRY_RUN:
        logger.info(f"GVHMR results already exist at {hmr4d_results_path}. Skipping pose estimation.")
        # Check if VitPose frames also exist
        if os.path.exists(vitpose_frames_dir):
            vitpose_frame_count = len([f for f in os.listdir(vitpose_frames_dir) if f.endswith('.jpg')])
            logger.info(f"VitPose frames directory exists with {vitpose_frame_count} frames at {vitpose_frames_dir}")
        return
    
    # Change to GVHMR directory for execution
    gvhmr_dir = "/home/mani/GVHMR"
    command = f'python tools/demo/demo.py --video "{video_path}" --output_root "{output_root}"'
    
    try:
        run_command(command, env_name=GVHMR_ENV, cwd=gvhmr_dir)
        logger.info(f"GVHMR pose estimation completed. Output in {output_root}")
        
        # Log information about created VitPose frames
        if os.path.exists(vitpose_frames_dir) and not DRY_RUN:
            vitpose_frame_count = len([f for f in os.listdir(vitpose_frames_dir) if f.endswith('.jpg')])
            logger.info(f"GVHMR created {vitpose_frame_count} VitPose overlay frames in {vitpose_frames_dir}")
        elif DRY_RUN:
            logger.info(f"DRY RUN: GVHMR would create VitPose frames in {vitpose_frames_dir}")
            
    except subprocess.CalledProcessError as e:
        logger.error(f"GVHMR pose estimation failed: {e}")
        raise


def run_gvhmr_to_hml3d_conversion(test_name, pose_output_path):
    """Convert GVHMR output to HumanML3D format."""
    logger.info(f"Converting GVHMR output to HumanML3D format for {test_name}")
    
    # Check if HumanML3D output already exists
    humanml3d_output_path = f"/home/mani/Repos/hcdt/data/humanml3d/{test_name}.pt"
    
    if os.path.exists(humanml3d_output_path) and not DRY_RUN:
        logger.info(f"HumanML3D file already exists at {humanml3d_output_path}. Skipping conversion.")
        return
    
    # Change to SMPLest-X directory for execution
    smplestx_dir = "/home/mani/SMPLest-X"
    command = f'python main/gvhmr_to_hml3d.py --test_name "{test_name}" --pose_output_path "{pose_output_path}"'
    
    try:
        run_command(command, env_name=SMPLESTX_ENV, cwd=smplestx_dir)
        logger.info(f"GVHMR to HumanML3D conversion completed for {test_name}")
    except subprocess.CalledProcessError as e:
        logger.error(f"GVHMR to HumanML3D conversion failed: {e}")
        raise


def run_gazelle_gaze_estimation(input_dir, output_dir):
    """Run Gazelle gaze estimation."""
    logger.info(f"Running Gazelle gaze estimation for {input_dir}")
    
    # Check if Gazelle results already exist
    gazelle_results_file = os.path.join(output_dir, "gaze_pixel_locations.txt")
    
    if os.path.exists(gazelle_results_file) and not DRY_RUN:
        logger.info(f"Gazelle results already exist at {gazelle_results_file}. Skipping gaze estimation.")
        return
    
    # Change to gazelle directory for execution
    gazelle_dir = "/home/mani/gazelle"
    command = f'python scripts/test2.py --input_dir "{input_dir}" --output_dir "{output_dir}"'
    
    try:
        run_command(command, env_name=GAZELLE_ENV, cwd=gazelle_dir)
        logger.info(f"Gazelle gaze estimation completed. Output in {output_dir}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Gazelle gaze estimation failed: {e}")
        raise


def process_video_set(input_folder, video_names, egocentric_video_names, set_name, args):
    """Process a set of videos (test or example)."""
    logger.info(f"Processing {set_name} video set in {input_folder}")
    
    # Process all videos for frame extraction
    all_videos = video_names + egocentric_video_names
    for video_name in all_videos:
        video_path = os.path.join(input_folder, video_name)
        
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            continue
        
        logger.info(f"Extracting frames from video: {video_name}")
        
        # Extract frames for all videos
        frames_dir = os.path.join(input_folder, "frames")
        if not (args and args.skip_frames):
            extract_frames_ffmpeg(video_path, frames_dir)
        else:
            logger.info("Skipping frame extraction (--skip-frames)")
    
    # Process only the first video for GVHMR
    if video_names:
        first_video = video_names[0]
        video_path = os.path.join(input_folder, first_video)
        
        if os.path.exists(video_path):
            logger.info(f"Running GVHMR pose estimation for first video: {first_video}")
            
            # Run GVHMR pose estimation only for first video
            gvhmr_output_root = os.path.join(input_folder, "GVHMR")
            if not (args and args.skip_gvhmr):
                run_gvhmr_pose_estimation(video_path, gvhmr_output_root)
            else:
                logger.info("Skipping GVHMR pose estimation (--skip-gvhmr)")
            
            # Extract frames from VitPose video overlay (if it exists)
            video_stem = Path(first_video).stem
            if not (args and args.skip_frames):
                extract_vitpose_overlay_frames(gvhmr_output_root, video_stem)
            else:
                logger.info("Skipping VitPose overlay frame extraction (--skip-frames)")
            
            # Convert to HumanML3D format (only for first video)
            hmr4d_results_path = os.path.join(gvhmr_output_root, video_stem, "hmr4d_results.pt")
            
            if set_name == "test":
                test_name = f"test_{video_stem}"
            else:  # example
                test_name = f"example_{video_stem}"
            
            if not (args and args.skip_conversion):
                if DRY_RUN or os.path.exists(hmr4d_results_path):
                    run_gvhmr_to_hml3d_conversion(test_name, hmr4d_results_path)
                else:
                    logger.warning(f"hmr4d_results.pt not found at {hmr4d_results_path}")
            else:
                logger.info("Skipping GVHMR to HumanML3D conversion (--skip-conversion)")
            
            # Create phase2 ground truth file (only for first video)
            vitpose_file_path = os.path.join(gvhmr_output_root, video_stem, "preprocess", "vitpose.pt")
            phase2_gt_output_path = os.path.join("/home/mani/Repos/hcdt/data/HAViD", f"phase2_{set_name}_{video_stem}.json")
            
            if not (args and args.skip_phase2_gt):
                # Try to read fps from config, default to 30 if not available
                fps = 30  # Default value
                try:
                    config_file_path = f"{config_path}{exp_name}.yaml"
                    if os.path.exists(config_file_path):
                        with open(config_file_path, 'r') as f:
                            config_data = yaml.safe_load(f)
                            fps = config_data.get('fps', 30)
                            logger.info(f"Using fps={fps} from config file")
                except Exception as e:
                    logger.warning(f"Could not read fps from config: {e}. Using default fps=30")
                
                create_phase2_ground_truth(vitpose_file_path, phase2_gt_output_path, fps)
            else:
                logger.info("Skipping phase2 ground truth generation (--skip-phase2-gt)")
    
    # Run Gazelle gaze estimation for all videos (using combined frames)
    frames_dir = os.path.join(input_folder, "frames")
    gazelle_output_dir = os.path.join(input_folder, "gazelle_output")
    if not (args and args.skip_gazelle):
        run_gazelle_gaze_estimation(frames_dir, gazelle_output_dir)
    else:
        logger.info("Skipping Gazelle gaze estimation (--skip-gazelle)")


def verify_prerequisites():
    """Verify that all required files and environments exist."""
    logger.info("Verifying prerequisites...")
    
    # Check conda environments
    envs_to_check = [GVHMR_ENV, SMPLESTX_ENV, GAZELLE_ENV]
    for env in envs_to_check:
        try:
            result = subprocess.run(f"conda info --envs | grep {env}", 
                                 shell=True, capture_output=True, text=True)
            if env not in result.stdout:
                logger.error(f"Conda environment '{env}' not found")
                return False
        except Exception as e:
            logger.error(f"Error checking conda environment {env}: {e}")
            return False
    
    # Check external script paths
    scripts_to_check = [GVHMR_DEMO_PATH, GVHMR_TO_HML3D_PATH, GAZELLE_SCRIPT_PATH]
    for script_path in scripts_to_check:
        if not os.path.exists(script_path):
            logger.error(f"Required script not found: {script_path}")
            return False
    
    # Check ffmpeg
    try:
        subprocess.run("ffmpeg -version", shell=True, capture_output=True, check=True)
    except subprocess.CalledProcessError:
        logger.error("ffmpeg not found or not working")
        return False
    
    logger.info("All prerequisites verified successfully")
    return True


def main():
    """Main processing pipeline."""
    parser = argparse.ArgumentParser(description='Process videos for Phase 2 experiments')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be done without actually running commands')
    parser.add_argument('--skip-frames', action='store_true',
                       help='Skip frame extraction (if frames already exist)')
    parser.add_argument('--skip-gvhmr', action='store_true',
                       help='Skip GVHMR pose estimation')
    parser.add_argument('--skip-conversion', action='store_true',
                       help='Skip GVHMR to HumanML3D conversion')
    parser.add_argument('--skip-gazelle', action='store_true',
                       help='Skip Gazelle gaze estimation')
    parser.add_argument('--skip-phase2-gt', action='store_true',
                       help='Skip phase2 ground truth file generation')
    
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No actual processing will be performed")
        global DRY_RUN
        DRY_RUN = True
    else:
        DRY_RUN = False
    
    logger.info(f"Starting preprocessing pipeline for experiment: {exp_name}")
    
    # Verify prerequisites
    if not verify_prerequisites():
        logger.error("Prerequisites check failed. Aborting.")
        sys.exit(1)
    
    try:
        # Create directory structure
        create_directories()
        
        # Generate directory configuration YAML
        dirs_config = generate_dirs_yaml()
        
        # Process example videos
        if example_video_names or example_egocentric_video_names:
            process_video_set(example_input_folder, example_video_names, example_egocentric_video_names, "example", args)
        
        # Process test videos
        if test_video_names or test_egocentric_video_names:
            process_video_set(test_input_folder, test_video_names, test_egocentric_video_names, "test", args)
        
        logger.info("Preprocessing pipeline completed successfully!")
        logger.info(f"Configuration file available at: {config_path}{exp_name}.yaml")
        logger.info(f"Directory configuration available at: {config_path}{exp_name}_dirs.yaml")
        logger.info("Generated phase2 ground truth files are available in: /home/mani/Repos/hcdt/data/HAViD/")
        logger.info("You can now run the phase2 experiment using the generated data.")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


def create_phase2_ground_truth(vitpose_file_path, output_file_path, fps=30):
    """
    Create phase2 ground truth file using VitPose data.
    
    Args:
        vitpose_file_path (str): Path to the VitPose .pt file
        output_file_path (str): Path where the phase2 ground truth JSON will be saved
        fps (int): Frames per second of the video (default: 30)
    """
    logger.info(f"Creating phase2 ground truth file from {vitpose_file_path}")
    
    if not os.path.exists(vitpose_file_path) and not DRY_RUN:
        logger.warning(f"VitPose file not found: {vitpose_file_path}. Skipping phase2 ground truth creation.")
        return
    
    if os.path.exists(output_file_path) and not DRY_RUN:
        logger.info(f"Phase2 ground truth file already exists at {output_file_path}. Skipping creation.")
        return
    
    if DRY_RUN:
        logger.info(f"DRY RUN: Would create phase2 ground truth file at {output_file_path}")
        return
    
    try:
        # Get total frames by loading the VitPose data
        import torch
        vitpose_data = torch.load(vitpose_file_path)
        total_frames = vitpose_data.shape[0]
        logger.info(f"VitPose data has {total_frames} frames")
        
        # Calculate frame range: start from frame 1+fps*4 (frame 121 for 30fps), increment by 60
        start_frame = 1 + fps * 4  # Frame 121 for 30fps (1-indexed)
        frame_increment = 60
        
        ground_truth_data = []
        
        # Generate ground truth entries
        current_frame = start_frame
        while current_frame <= total_frames:
            # Convert to 0-indexed for the function call
            frame_index = current_frame - 1
            
            try:
                # Get hand positions from VitPose data
                left_hand_x, left_hand_y, right_hand_x, right_hand_y = get_hand_xy_positions(
                    vitpose_file_path, frame_index
                )
                
                # Create ground truth entry (similar structure to phase2_simplified.json)
                gt_entry = {
                    "frame": current_frame,
                    "actual_hand_positions": {
                        "left_hand_x": left_hand_x,
                        "left_hand_y": left_hand_y, 
                        "right_hand_x": right_hand_x,
                        "right_hand_y": right_hand_y
                    }
                }
                
                ground_truth_data.append(gt_entry)
                logger.debug(f"Added ground truth for frame {current_frame}")
                
            except Exception as e:
                logger.warning(f"Failed to get hand positions for frame {current_frame}: {e}")
            
            current_frame += frame_increment
        
        # Save to JSON file
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, 'w') as f:
            json.dump(ground_truth_data, f, indent=2)
        
        logger.info(f"Created phase2 ground truth file with {len(ground_truth_data)} entries at {output_file_path}")
        logger.info(f"Frame range: {start_frame} to {current_frame - frame_increment} (increment: {frame_increment})")
        logger.info(f"Total frames in VitPose data: {total_frames}")
        
    except Exception as e:
        logger.error(f"Failed to create phase2 ground truth file: {e}")
        raise


if __name__ == "__main__":
    main()



