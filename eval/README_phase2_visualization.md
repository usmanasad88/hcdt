# Phase 2 Visualization Tools

This directory contains tools for visualizing and analyzing the results of Phase 2 experiments, which predict hand positions based on video sequences.

## Overview

Phase 2 experiments take a sequence of 10 images (0.2 seconds apart) with human pose overlays and hand position data, then predict where the left and right hands will be positioned 2 seconds after the last input frame.

## Tools

### 1. Main Visualization Tool (`visualize_phase_two.py`)

The comprehensive visualization tool that provides full control over the visualization process.

**Features:**
- Load and process phase_two experiment results
- Overlay predicted hand positions on VitPose frames  
- Show ground truth positions for comparison
- Calculate prediction errors and generate analysis plots
- Create videos with prediction overlays
- Generate comprehensive summary reports

**Usage:**
```bash
# Basic usage
python eval/visualize_phase_two.py logs/phase2_results.json

# Advanced options
python eval/visualize_phase_two.py logs/phase2_results.json \
    --config config/visualization_config.yaml \
    --no-ground-truth \
    --no-video \
    --output-dir custom_output
```

**Output Structure:**
```
outputs/phase2_visualization_TIMESTAMP/
├── frames_with_predictions/     # Individual frames with overlays
├── comparison_frames/           # Side-by-side comparisons
├── videos/                      # Generated videos
├── analysis/                    # Error analysis plots and stats
└── summary_report.txt          # Comprehensive summary
```

### 2. Quick Visualization (`quick_visualize_phase2.py`)

A simplified interface for rapid visualization with preset modes.

**Usage:**
```bash
# Auto-detect latest results and run full visualization
./eval/quick_visualize_phase2.py

# Specify file and mode
./eval/quick_visualize_phase2.py logs/phase2_results.json --mode analysis

# List available results files
./eval/quick_visualize_phase2.py --list
```

**Modes:**
- `quick`: Just frame overlays, no video or analysis
- `analysis`: Frames + error analysis, no video  
- `video`: Frames + video generation, no analysis
- `full`: Complete visualization pipeline (default)

### 3. Batch Processor (`batch_process_phase2.py`)

Process multiple phase2 results files for comparison analysis.

**Usage:**
```bash
# Process all phase2 files in logs directory
./eval/batch_process_phase2.py

# Process specific files
./eval/batch_process_phase2.py --files logs/phase2_exp1.json logs/phase2_exp2.json

# List available files
./eval/batch_process_phase2.py --list
```

**Output:**
- Individual visualizations for each file
- Comparison report across all experiments
- CSV data for further analysis

## Configuration

Visualization behavior can be customized using `config/visualization_config.yaml`:

```yaml
visualization:
  colors:
    left_hand_prediction: "red"      # Color for left hand predictions
    right_hand_prediction: "blue"    # Color for right hand predictions
    left_hand_ground_truth: "green"  # Color for left hand ground truth
    right_hand_ground_truth: "orange" # Color for right hand ground truth
  
  video:
    fps: 15                          # Video frame rate
    pause_duration: 2                # Pause duration at key frames
  
  overlays:
    show_frame_number: true          # Show frame numbers
    show_target_object: true         # Show target object names
    show_reasoning: true             # Show reasoning text
    max_reasoning_length: 60         # Truncate long reasoning text
```

## Visualization Elements

### Frame Overlays
- **Red circles**: Left hand predictions
- **Blue circles**: Right hand predictions  
- **Green circles**: Left hand ground truth (if available)
- **Orange circles**: Right hand ground truth (if available)
- **Text overlays**: Frame number, target object, reasoning summary

### Error Analysis
- **Error over time plots**: Track prediction accuracy across frames
- **Error distribution histograms**: Show overall accuracy patterns
- **Box plots**: Compare left vs right hand accuracy
- **Statistics tables**: Mean, std dev, min/max errors

### Video Output
- **Prediction videos**: Show predictions overlaid on original frames
- **Pause functionality**: Automatic pauses at key prediction frames
- **Multiple formats**: Support for different video codecs

## Requirements

```bash
pip install pillow matplotlib numpy pandas omegaconf
```

## Examples

### Visualize Latest Results
```bash
# Quick visualization of the most recent experiment
./eval/quick_visualize_phase2.py --mode full
```

### Compare Multiple Experiments
```bash
# Process all experiments for comparison
./eval/batch_process_phase2.py --directory logs/

# View the comparison report
cat outputs/phase2_batch_*/batch_comparison_report.txt
```

### Custom Visualization
```python
from eval.visualize_phase_two import Phase2Visualizer

# Create visualizer with custom config
visualizer = Phase2Visualizer("config/custom_visualization.yaml")

# Load results
visualizer.load_results("logs/my_experiment.json")

# Run specific parts of the pipeline
visualizer.create_output_directory("my_custom_analysis")
visualizer.visualize_predictions_on_frames(show_ground_truth=True)
errors = visualizer.calculate_prediction_errors()
visualizer.create_error_analysis_plots(errors)
```

## Troubleshooting

### Common Issues

1. **No VitPose frames found**
   - Check that `test_vitpose_frames` path in config is correct
   - Ensure frames are named with consistent pattern (e.g., `frame_0001.png`)

2. **Missing ground truth data**
   - Some results files may not include `actual_hand_positions`
   - Use `--no-ground-truth` flag to skip GT comparison

3. **Video creation fails**
   - Ensure ffmpeg is installed and in PATH
   - Check that frame directory contains valid image files

4. **Font rendering issues**
   - Install system fonts or specify font path in config
   - Falls back to default fonts if system fonts unavailable

### File Format Requirements

**Phase 2 Results JSON Format:**
```json
[
  {
    "input_end_frame": 256,
    "prediction_frame": 286,
    "predicted_hand_positions": {
      "left_hand_x": 550.0,
      "left_hand_y": 550.0,
      "right_hand_x": 500.0,
      "right_hand_y": 500.0
    },
    "actual_hand_positions": {
      "left_hand_x": 612.58,
      "left_hand_y": 512.19,
      "right_hand_x": 517.44,
      "right_hand_y": 563.63
    },
    "reasoning_summary": "Brief reasoning text",
    "target_object": "Object being manipulated"
  }
]
```

## Integration with Existing Tools

These visualization tools integrate with:
- **run_phase_two.py**: Automatic result loading
- **motionutils.py**: Hand position data extraction
- **create_vid.py**: Video generation utilities
- **overlay_genai.py**: Advanced overlay functionality

For more information, see the main project documentation.
