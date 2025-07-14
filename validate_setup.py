#!/usr/bin/env python3
"""
Quick validation script to check if the setup is ready for process_task.py
"""

import os
import subprocess
from pathlib import Path

def check_videos():
    """Check if input videos exist."""
    print("Checking input videos...")
    
    test_video = "/home/mani/Central/Stack/exp2/cam01.mp4"
    example_video = "/home/mani/Central/Stack/cam1/cam01.mp4"
    
    videos = [
        ("Test video", test_video),
        ("Example video", example_video)
    ]
    
    for name, path in videos:
        if os.path.exists(path):
            print(f"✓ {name}: {path}")
        else:
            print(f"✗ {name}: {path} - NOT FOUND")

def check_conda_envs():
    """Check if required conda environments exist."""
    print("\nChecking conda environments...")
    
    envs = ["gvhmr", "smplestx", "gazelle"]
    
    try:
        result = subprocess.run("conda info --envs", shell=True, capture_output=True, text=True)
        env_list = result.stdout
        
        for env in envs:
            if env in env_list:
                print(f"✓ Conda environment: {env}")
            else:
                print(f"✗ Conda environment: {env} - NOT FOUND")
    except Exception as e:
        print(f"✗ Error checking conda environments: {e}")

def check_external_scripts():
    """Check if external scripts exist."""
    print("\nChecking external scripts...")
    
    scripts = [
        ("GVHMR demo", "/home/mani/GVHMR/tools/demo/demo.py"),
        ("GVHMR to HML3D", "/home/mani/SMPLest-X/main/gvhmr_to_hml3d.py"),
        ("Gazelle script", "/home/mani/gazelle/scripts/test2.py")
    ]
    
    for name, path in scripts:
        if os.path.exists(path):
            print(f"✓ {name}: {path}")
        else:
            print(f"✗ {name}: {path} - NOT FOUND")

def check_ffmpeg():
    """Check if ffmpeg is available."""
    print("\nChecking ffmpeg...")
    
    try:
        result = subprocess.run("ffmpeg -version", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ ffmpeg is available")
        else:
            print("✗ ffmpeg failed to run")
    except Exception as e:
        print(f"✗ ffmpeg error: {e}")

def main():
    print("=== Process Task Validation ===\n")
    
    check_videos()
    check_conda_envs()
    check_external_scripts()
    check_ffmpeg()
    
    print("\n=== Validation Complete ===")
    print("If any items show ✗, please fix them before running process_task.py")

if __name__ == "__main__":
    main()
