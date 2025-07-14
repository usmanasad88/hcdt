#!/usr/bin/env python3
"""
Quick visualization script for Phase 2 experiment results.
Usage: python quick_visualize_phase2.py [results_file] [options]
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eval.visualize_phase_two import Phase2Visualizer


def find_latest_phase2_results(logs_dir: str = "logs") -> str:
    """Find the most recent phase2 results file."""
    if not os.path.exists(logs_dir):
        return None
    
    phase2_files = [f for f in os.listdir(logs_dir) if 'phase2' in f.lower() and f.endswith('.json')]
    if not phase2_files:
        return None
    
    # Sort by modification time
    phase2_files.sort(key=lambda x: os.path.getmtime(os.path.join(logs_dir, x)), reverse=True)
    return os.path.join(logs_dir, phase2_files[0])


def quick_visualize(results_file: str = None, mode: str = "full"):
    """Quick visualization with preset modes."""
    
    # Auto-find results file if not provided
    if not results_file:
        results_file = find_latest_phase2_results()
        if not results_file:
            print("❌ No phase2 results file found. Please specify one.")
            return False
        print(f"📄 Auto-detected results file: {results_file}")
    
    # Check if file exists
    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        return False
    
    # Create visualizer
    visualizer = Phase2Visualizer()
    
    # Configure based on mode
    if mode == "quick":
        # Quick mode: just frames, no video, no analysis
        output_dir = visualizer.run_complete_visualization(
            results_file=results_file,
            show_ground_truth=True,
            create_video=False,
            analyze_errors=False
        )
    elif mode == "analysis":
        # Analysis mode: frames + error analysis, no video
        output_dir = visualizer.run_complete_visualization(
            results_file=results_file,
            show_ground_truth=True,
            create_video=False,
            analyze_errors=True
        )
    elif mode == "video":
        # Video mode: frames + video, no analysis
        output_dir = visualizer.run_complete_visualization(
            results_file=results_file,
            show_ground_truth=True,
            create_video=True,
            analyze_errors=False
        )
    else:  # mode == "full"
        # Full mode: everything
        output_dir = visualizer.run_complete_visualization(
            results_file=results_file,
            show_ground_truth=True,
            create_video=True,
            analyze_errors=True
        )
    
    return output_dir is not None


def main():
    parser = argparse.ArgumentParser(description="Quick Phase 2 visualization")
    parser.add_argument("results_file", nargs="?", help="Path to phase2 results JSON file (auto-detected if not provided)")
    parser.add_argument("--mode", choices=["quick", "analysis", "video", "full"], default="full",
                       help="Visualization mode (default: full)")
    parser.add_argument("--list", action="store_true", help="List available phase2 results files")
    
    args = parser.parse_args()
    
    if args.list:
        logs_dir = "logs"
        if os.path.exists(logs_dir):
            phase2_files = [f for f in os.listdir(logs_dir) if 'phase2' in f.lower() and f.endswith('.json')]
            if phase2_files:
                print("Available phase2 results files:")
                for f in sorted(phase2_files):
                    file_path = os.path.join(logs_dir, f)
                    mtime = os.path.getmtime(file_path)
                    mtime_str = __import__('datetime').datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                    print(f"  - {f} (modified: {mtime_str})")
            else:
                print("No phase2 results files found in logs directory.")
        else:
            print("Logs directory not found.")
        return
    
    success = quick_visualize(args.results_file, args.mode)
    if success:
        print("\n✅ Quick visualization completed successfully!")
    else:
        print("\n❌ Quick visualization failed.")


if __name__ == "__main__":
    main()
