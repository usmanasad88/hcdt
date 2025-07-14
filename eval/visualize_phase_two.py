"""
Streamlined Phase 2 Experiment Visualization Tool

This script provides a comprehensive visualization pipeline for phase_two experiments,
which predict hand positions based on video sequences. 

Features:
- Load and process phase_two experiment results
- Overlay predicted hand positions on VitPose frames
- Create comparison videos showing predictions vs ground truth
- Generate summary statistics and error analysis
- Support for multiple visualization modes
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.imgutils import draw_point_on_image, draw_multiple_points_on_image, image_to_base64
from utils.motionutils import get_hand_xy_positions
from utils.create_vid import create_video_with_pauses
from utils.overlay_genai import overlay_genai_video_phase2
from omegaconf import DictConfig, OmegaConf


class Phase2Visualizer:
    """Main class for visualizing phase_two experiment results."""
    
    def __init__(self, config_path: str = None):
        """Initialize the visualizer with configuration."""
        self.config = self._load_config(config_path)
        self.results_data = None
        self.output_dir = None
        self.base_frames_dir = self.config.exp.test_vitpose_frames if 'test_vitpose_frames' in self.config.exp else None
        
    def _load_config(self, config_path: str = None) -> DictConfig:
        """Load configuration from file or use defaults."""
        # Priority: user config > visualization config > default config > minimal fallback
        config_paths = []
        if config_path and os.path.exists(config_path):
            config_paths.append(config_path)
        vis_config = os.path.join(os.path.dirname(__file__), '..', 'config', 'visualization_config.yaml')
        if os.path.exists(vis_config):
            config_paths.append(vis_config)
        default_config = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        if os.path.exists(default_config):
            config_paths.append(default_config)
        for path in config_paths:
            try:
                return OmegaConf.load(path)
            except Exception as e:
                print(f"Warning: Could not load config from {path}: {e}")
        print("Warning: No config file found. Using minimal defaults.")
        return OmegaConf.create({
            'exp': {
                'test_vitpose_frames': '/home/mani/Central/Stack/exp2/GVHMR/cam01/preprocess/vitpose_overlay_frames',
                'frame_width': 1920,
                'frame_height': 1080,
                'fps': 30
            },
            'visualization': {
                'output_base_dir': 'outputs',
                'create_subdirs': True,
                'point_radius': 15,
                'font_size': 20,
                'small_font_size': 16
            }
        })
    
    def load_results(self, results_file: str) -> bool:
        """Load phase_two experiment results from JSON file."""
        try:
            with open(results_file, 'r') as f:
                self.results_data = json.load(f)
            print(f"✅ Loaded {len(self.results_data)} prediction results from {results_file}")
            return True
        except FileNotFoundError:
            print(f"❌ Results file not found: {results_file}")
            return False
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing JSON results: {e}")
            return False
    
    def create_output_directory(self, base_name: str = "phase2_visualization") -> str:
        """Create output directory for visualization results."""
        timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join("outputs", f"{base_name}_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create subdirectories
        os.makedirs(os.path.join(self.output_dir, "frames_with_predictions"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "comparison_frames"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "videos"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "analysis"), exist_ok=True)
        
        print(f"📁 Created output directory: {self.output_dir}")
        return self.output_dir
    
    def visualize_predictions_on_frames(self, show_ground_truth: bool = True) -> str:
        """
        Create frames with predicted hand positions overlaid.
        
        Args:
            show_ground_truth: Whether to also show ground truth positions
            
        Returns:
            Path to the directory containing visualization frames
        """
        if not self.results_data:
            print("❌ No results data loaded. Call load_results() first.")
            return None
            
        frames_dir = os.path.join(self.output_dir, "frames_with_predictions")
        
        # Get available frame files
        vitpose_frames_dir = self.config.exp.test_vitpose_frames
        if not os.path.exists(vitpose_frames_dir):
            print(f"❌ VitPose frames directory not found: {vitpose_frames_dir}")
            return None
            
        frame_files = sorted([f for f in os.listdir(vitpose_frames_dir) if f.endswith('.png')])
        
        for entry in self.results_data:
            prediction_frame = entry.get("prediction_frame", entry.get("frame"))
            if prediction_frame is None:
                continue
                
            # Find corresponding frame file
            frame_file = None
            for f in frame_files:
                if f"frame_{prediction_frame:04d}" in f or f"{prediction_frame:04d}" in f:
                    frame_file = f
                    break
            
            if not frame_file:
                print(f"⚠️  Frame file not found for prediction frame {prediction_frame}")
                continue
                
            frame_path = os.path.join(vitpose_frames_dir, frame_file)
            output_path = os.path.join(frames_dir, f"frame_{prediction_frame:06d}.png")
            
            # Extract prediction data
            if "predicted_hand_positions" in entry:
                pred_positions = entry["predicted_hand_positions"]
            elif "predicted_positions" in entry:
                pred_positions = entry["predicted_positions"]
            else:
                print(f"⚠️  No prediction positions found for frame {prediction_frame}")
                continue
            
            # Create visualization
            self._draw_prediction_on_frame(
                frame_path, 
                output_path, 
                pred_positions, 
                entry.get("actual_hand_positions") if show_ground_truth else None,
                frame_number=prediction_frame,
                reasoning=entry.get("reasoning", entry.get("reasoning_summary", "")),
                target_object=entry.get("target_object", "")
            )
        
        print(f"✅ Created {len(self.results_data)} prediction visualization frames")
        return frames_dir
    
    def _draw_prediction_on_frame(self, frame_path: str, output_path: str, 
                                 predictions: Dict, ground_truth: Dict = None,
                                 frame_number: int = 0, reasoning: str = "", 
                                 target_object: str = ""):
        """Draw predictions and optionally ground truth on a single frame."""
        img = Image.open(frame_path).convert("RGB")
        width, height = img.size
        draw = ImageDraw.Draw(img)
        
        # Load font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except (OSError, IOError):
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Draw predicted positions (Red for left, Blue for right)
        point_radius = 15
        
        # Left hand prediction (Red)
        if "left_hand_x" in predictions and "left_hand_y" in predictions:
            left_x_px = int((predictions["left_hand_x"] / 1000.0) * width)
            left_y_px = int((predictions["left_hand_y"] / 1000.0) * height)
            
            draw.ellipse([
                (left_x_px - point_radius, left_y_px - point_radius),
                (left_x_px + point_radius, left_y_px + point_radius)
            ], fill='red', outline='darkred', width=3)
            draw.text((left_x_px + 20, left_y_px - 10), "L_pred", fill='red', font=small_font)
        
        # Right hand prediction (Blue)
        if "right_hand_x" in predictions and "right_hand_y" in predictions:
            right_x_px = int((predictions["right_hand_x"] / 1000.0) * width)
            right_y_px = int((predictions["right_hand_y"] / 1000.0) * height)
            
            draw.ellipse([
                (right_x_px - point_radius, right_y_px - point_radius),
                (right_x_px + point_radius, right_y_px + point_radius)
            ], fill='blue', outline='darkblue', width=3)
            draw.text((right_x_px + 20, right_y_px - 10), "R_pred", fill='blue', font=small_font)
        
        # Draw ground truth positions if available (Green for left, Orange for right)
        if ground_truth:
            gt_radius = 12
            
            # Left hand ground truth (Green)
            if "left_hand_x" in ground_truth and "left_hand_y" in ground_truth:
                left_gt_x_px = int((ground_truth["left_hand_x"] / 1000.0) * width)
                left_gt_y_px = int((ground_truth["left_hand_y"] / 1000.0) * height)
                
                draw.ellipse([
                    (left_gt_x_px - gt_radius, left_gt_y_px - gt_radius),
                    (left_gt_x_px + gt_radius, left_gt_y_px + gt_radius)
                ], fill='green', outline='darkgreen', width=2)
                draw.text((left_gt_x_px + 20, left_gt_y_px + 15), "L_gt", fill='green', font=small_font)
            
            # Right hand ground truth (Orange)
            if "right_hand_x" in ground_truth and "right_hand_y" in ground_truth:
                right_gt_x_px = int((ground_truth["right_hand_x"] / 1000.0) * width)
                right_gt_y_px = int((ground_truth["right_hand_y"] / 1000.0) * height)
                
                draw.ellipse([
                    (right_gt_x_px - gt_radius, right_gt_y_px - gt_radius),
                    (right_gt_x_px + gt_radius, right_gt_y_px + gt_radius)
                ], fill='orange', outline='darkorange', width=2)
                draw.text((right_gt_x_px + 20, right_gt_y_px + 15), "R_gt", fill='orange', font=small_font)
        
        # Add frame information and metadata
        info_y = 30
        draw.text((10, info_y), f"Frame {frame_number} - Prediction Visualization", 
                 fill='white', font=font)
        
        if target_object:
            info_y += 35
            draw.text((10, info_y), f"Target: {target_object}", fill='yellow', font=small_font)
        
        if reasoning:
            info_y += 25
            # Truncate reasoning if too long
            reasoning_text = reasoning[:60] + "..." if len(reasoning) > 60 else reasoning
            draw.text((10, info_y), f"Reasoning: {reasoning_text}", fill='lightgray', font=small_font)
        
        # Add legend
        legend_y = height - 120
        draw.text((10, legend_y), "Legend:", fill='white', font=small_font)
        draw.text((10, legend_y + 20), "🔴 Left Hand Prediction", fill='red', font=small_font)
        draw.text((10, legend_y + 40), "🔵 Right Hand Prediction", fill='blue', font=small_font)
        if ground_truth:
            draw.text((10, legend_y + 60), "🟢 Left Hand Ground Truth", fill='green', font=small_font)
            draw.text((10, legend_y + 80), "🟠 Right Hand Ground Truth", fill='orange', font=small_font)
        
        img.save(output_path)
    
    def calculate_prediction_errors(self) -> Dict[str, Any]:
        """Calculate prediction errors and generate analysis."""
        if not self.results_data:
            print("❌ No results data loaded.")
            return {}
        
        errors = {
            'left_hand_errors': [],
            'right_hand_errors': [],
            'combined_errors': [],
            'frame_numbers': [],
            'target_objects': []
        }
        
        for entry in self.results_data:
            if "predicted_hand_positions" not in entry and "predicted_positions" not in entry:
                continue
            if "actual_hand_positions" not in entry:
                continue
                
            pred = entry.get("predicted_hand_positions", entry.get("predicted_positions", {}))
            actual = entry["actual_hand_positions"]
            
            # Calculate Euclidean distances (in normalized 0-1000 space)
            if all(k in pred and k in actual for k in ['left_hand_x', 'left_hand_y']):
                left_error = np.sqrt(
                    (pred['left_hand_x'] - actual['left_hand_x'])**2 + 
                    (pred['left_hand_y'] - actual['left_hand_y'])**2
                )
                errors['left_hand_errors'].append(left_error)
            
            if all(k in pred and k in actual for k in ['right_hand_x', 'right_hand_y']):
                right_error = np.sqrt(
                    (pred['right_hand_x'] - actual['right_hand_x'])**2 + 
                    (pred['right_hand_y'] - actual['right_hand_y'])**2
                )
                errors['right_hand_errors'].append(right_error)
            
            # Combined error (average of both hands)
            if len(errors['left_hand_errors']) == len(errors['right_hand_errors']) + 1:
                combined_error = (errors['left_hand_errors'][-1] + errors['right_hand_errors'][-1]) / 2
                errors['combined_errors'].append(combined_error)
            
            errors['frame_numbers'].append(entry.get('prediction_frame', entry.get('frame', 0)))
            errors['target_objects'].append(entry.get('target_object', 'Unknown'))
        
        # Calculate statistics
        stats = {}
        for hand in ['left_hand_errors', 'right_hand_errors', 'combined_errors']:
            if errors[hand]:
                stats[hand] = {
                    'mean': np.mean(errors[hand]),
                    'std': np.std(errors[hand]),
                    'min': np.min(errors[hand]),
                    'max': np.max(errors[hand]),
                    'median': np.median(errors[hand])
                }
        
        errors['statistics'] = stats
        return errors
    
    def create_error_analysis_plots(self, errors: Dict) -> str:
        """Create plots for error analysis."""
        analysis_dir = os.path.join(self.output_dir, "analysis")
        
        # Error distribution plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Phase 2 Prediction Error Analysis', fontsize=16)
        
        # Left hand errors over frames
        if errors['left_hand_errors']:
            axes[0, 0].plot(errors['frame_numbers'][:len(errors['left_hand_errors'])], 
                           errors['left_hand_errors'], 'r-o', label='Left Hand')
            axes[0, 0].set_title('Left Hand Prediction Errors Over Frames')
            axes[0, 0].set_xlabel('Frame Number')
            axes[0, 0].set_ylabel('Error (normalized pixels)')
            axes[0, 0].grid(True)
        
        # Right hand errors over frames
        if errors['right_hand_errors']:
            axes[0, 1].plot(errors['frame_numbers'][:len(errors['right_hand_errors'])], 
                           errors['right_hand_errors'], 'b-o', label='Right Hand')
            axes[0, 1].set_title('Right Hand Prediction Errors Over Frames')
            axes[0, 1].set_xlabel('Frame Number')
            axes[0, 1].set_ylabel('Error (normalized pixels)')
            axes[0, 1].grid(True)
        
        # Error distribution histogram
        all_errors = errors['left_hand_errors'] + errors['right_hand_errors']
        if all_errors:
            axes[1, 0].hist(all_errors, bins=20, alpha=0.7, color='purple')
            axes[1, 0].set_title('Error Distribution')
            axes[1, 0].set_xlabel('Error (normalized pixels)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True)
        
        # Box plot comparison
        if errors['left_hand_errors'] and errors['right_hand_errors']:
            axes[1, 1].boxplot([errors['left_hand_errors'], errors['right_hand_errors']], 
                              labels=['Left Hand', 'Right Hand'])
            axes[1, 1].set_title('Error Comparison by Hand')
            axes[1, 1].set_ylabel('Error (normalized pixels)')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(analysis_dir, "error_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save error statistics to text file
        stats_path = os.path.join(analysis_dir, "error_statistics.txt")
        with open(stats_path, 'w') as f:
            f.write("Phase 2 Prediction Error Statistics\n")
            f.write("=" * 40 + "\n\n")
            
            for hand_type, stats in errors['statistics'].items():
                f.write(f"{hand_type.replace('_', ' ').title()}:\n")
                f.write(f"  Mean Error: {stats['mean']:.2f} normalized pixels\n")
                f.write(f"  Std Dev: {stats['std']:.2f}\n")
                f.write(f"  Min Error: {stats['min']:.2f}\n")
                f.write(f"  Max Error: {stats['max']:.2f}\n")
                f.write(f"  Median Error: {stats['median']:.2f}\n")
                f.write("\n")
        
        print(f"📊 Error analysis plots saved to {plot_path}")
        print(f"📈 Error statistics saved to {stats_path}")
        return analysis_dir
    
    def create_prediction_video(self, frame_rate: int = 15, pause_frames: List[int] = None) -> str:
        """Create a video from prediction frames."""
        frames_dir = os.path.join(self.output_dir, "frames_with_predictions")
        if not os.path.exists(frames_dir):
            print("❌ No prediction frames found. Run visualize_predictions_on_frames() first.")
            return None
        
        video_output = os.path.join(self.output_dir, "videos", "phase2_predictions.mp4")
        
        # Use the create_vid utility
        create_video_with_pauses(
            input_frames_dir=frames_dir,
            base_frames_dir=self.base_frames_dir,
            output_video_path=video_output,
            base_fps=frame_rate,
            pause_duration_seconds=2,
            initial_pause_frame=5,
            subsequent_pause_interval=10
        )
        
        print(f"🎥 Prediction video created: {video_output}")
        return video_output
    
    def generate_summary_report(self, errors: Dict) -> str:
        """Generate a comprehensive summary report."""
        report_path = os.path.join(self.output_dir, "summary_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("Phase 2 Experiment Visualization Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Results File: {getattr(self, 'results_file', 'Unknown')}\n")
            f.write(f"Total Predictions: {len(self.results_data)}\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")
            
            # Error summary
            if errors and 'statistics' in errors:
                f.write("Prediction Accuracy Summary:\n")
                f.write("-" * 30 + "\n")
                
                for hand_type, stats in errors['statistics'].items():
                    f.write(f"{hand_type.replace('_', ' ').title()}:\n")
                    f.write(f"  Average Error: {stats['mean']:.2f} normalized pixels\n")
                    f.write(f"  Error Range: {stats['min']:.2f} - {stats['max']:.2f}\n\n")
            
            # Target objects summary
            if hasattr(self, 'results_data') and self.results_data:
                target_objects = [entry.get('target_object', 'Unknown') for entry in self.results_data]
                unique_targets = list(set(target_objects))
                f.write(f"Target Objects Predicted: {len(unique_targets)}\n")
                for target in unique_targets:
                    count = target_objects.count(target)
                    f.write(f"  - {target}: {count} predictions\n")
            
            f.write(f"\nVisualization completed on: {__import__('datetime').datetime.now()}\n")
        
        print(f"📄 Summary report saved to {report_path}")
        return report_path
    
    def run_complete_visualization(self, results_file: str, show_ground_truth: bool = True,
                                 create_video: bool = True, analyze_errors: bool = True) -> str:
        """Run the complete visualization pipeline."""
        print("🚀 Starting Phase 2 visualization pipeline...")
        
        # Load results
        if not self.load_results(results_file):
            return None
        
        # Create output directory
        self.create_output_directory()
        self.results_file = results_file
        
        # Visualize predictions on frames
        frames_dir = self.visualize_predictions_on_frames(show_ground_truth)
        
        # Calculate errors and create analysis
        errors = {}
        if analyze_errors and show_ground_truth:
            errors = self.calculate_prediction_errors()
            if errors:
                self.create_error_analysis_plots(errors)
        
        # Create video
        if create_video:
            self.create_prediction_video()
        
        # Generate summary report
        self.generate_summary_report(errors)
        
        print(f"✅ Visualization pipeline completed successfully!")
        print(f"📁 All outputs saved to: {self.output_dir}")
        
        return self.output_dir

import datetime

def get_default_output_dir(base_name="phase2_visualization"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("logs", f"{base_name}_{timestamp}")

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Visualize Phase 2 experiment results")
    parser.add_argument("results_file", nargs="?", default="data/Stack/phase2_icl_result_window_3.json", help="Path to the phase_two results JSON file")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--no-ground-truth", action="store_true", 
                       help="Don't show ground truth positions")
    parser.add_argument("--no-video", action="store_true", 
                       help="Don't create output video")
    parser.add_argument("--no-analysis", action="store_true", 
                       help="Don't perform error analysis")
    parser.add_argument("--output-dir", default=get_default_output_dir(),
                        help="Override output directory name (default: logs/phase2_visualization_TIMESTAMP)")
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = Phase2Visualizer(args.config)
    
    # Override output directory if specified
    visualizer.output_dir = args.output_dir
    os.makedirs(visualizer.output_dir, exist_ok=True)
    
    # Run visualization
    result = visualizer.run_complete_visualization(
        results_file=args.results_file,
        show_ground_truth=not args.no_ground_truth,
        create_video=not args.no_video,
        analyze_errors=not args.no_analysis
    )
    
    if result:
        print(f"\n🎉 Visualization completed successfully!")
        print(f"📁 Results available at: {result}")
    else:
        print("❌ Visualization failed.")

if __name__ == "__main__":
    main()

