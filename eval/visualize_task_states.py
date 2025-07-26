import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def load_json_data(file_path):
    """Loads data from a JSON file with error handling."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' is not a valid JSON file.")
        return None

def process_ground_truth(data):
    """Processes the ground truth data to find start, end, and completion times."""
    action_timings = {}
    all_actions = set()
    max_frame = 0

    if not data:
        return {}, set(), 0

    for entry in data:
        frame = entry['frame_number']
        if frame > max_frame:
            max_frame = frame

        for action in entry['state'].get('steps_in_progress', []):
            if action == 'idle': continue
            all_actions.add(action)
            if action not in action_timings:
                action_timings[action] = {"start": frame, "end": None, "completed": None}

    previous_completed = set()
    for entry in data:
        frame = entry['frame_number']
        current_completed = set(entry['state'].get('steps_completed', []))

        newly_completed_actions = current_completed - previous_completed

        for action in newly_completed_actions:
            if action in action_timings:
                if action_timings[action]['completed'] is None:
                    action_timings[action]['completed'] = frame
                if action_timings[action]['end'] is None:
                    action_timings[action]['end'] = frame

        previous_completed = current_completed

    return action_timings, all_actions, max_frame

def process_predictions(data):
    """Processes prediction data to find frames where actions are marked."""
    predictions = defaultdict(lambda: {'in_progress': [], 'completed': []})
    all_actions = set()
    max_frame = 0

    if not data:
        return {}, set(), 0

    for entry in data:
        frame = entry['frame_number']
        if frame > max_frame:
            max_frame = frame

        state = entry.get('state', {})

        for action in state.get('steps_in_progress', []):
            if action == 'idle': continue
            all_actions.add(action)
            predictions[action]['in_progress'].append(frame)

        for action in state.get('steps_completed', []):
            if action == 'idle': continue
            all_actions.add(action)
            predictions[action]['completed'].append(frame)

    return predictions, all_actions, max_frame

def visualize_comparison_timeline(gt_file_path, pred_file_path, output_path=None, pred_bar_duration=30):
    """
    Visualizes and compares ground truth and prediction timelines for actions.

    Args:
        gt_file_path (str): Path to the ground truth JSON file.
        pred_file_path (str): Path to the prediction JSON file.
        output_path (str): Path to save the plot image. If None, the plot is displayed.
        pred_bar_duration (int): The visual duration (in frames) for each prediction bar.
    """
    gt_data = load_json_data(gt_file_path)
    pred_data = load_json_data(pred_file_path)

    if gt_data is None or pred_data is None:
        return

    gt_timings, gt_actions, gt_max_frame = process_ground_truth(gt_data)
    pred_timings, pred_actions, pred_max_frame = process_predictions(pred_data)

    # Combine all unique actions from both datasets
    combined_actions = list(gt_actions.union(pred_actions))

    # Sort actions based on their ground truth start time.
    # Actions only in predictions (without a GT start time) are placed at the end.
    all_actions = sorted(combined_actions, key=lambda action: gt_timings.get(action, {}).get('start', float('inf')))

    # Create a mapping from action name to its y-position on the chart
    action_y_map = {action: i for i, action in enumerate(all_actions)}

    max_frame = max(gt_max_frame, pred_max_frame)

    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(20, 12))

    # Plot Ground Truth Bars
    for action, timings in gt_timings.items():
        y_pos = action_y_map.get(action)
        if y_pos is None: continue

        # Plot GT 'in-progress' bar
        if timings.get('start') is not None and timings.get('end') is not None:
            duration = timings['end'] - timings['start']
            ax.barh(y_pos, duration, left=timings['start'], height=0.4, color='royalblue',
                    label='GT In Progress' if 'GT In Progress' not in [h.get_label() for h in ax.get_legend_handles_labels()[0]] else "")

        # Plot GT 'completed' bar
        if timings.get('completed') is not None:
            completed_duration = max_frame - timings['completed']
            ax.barh(y_pos, completed_duration, left=timings['completed'], height=0.4,
                    color='mediumseagreen', alpha=0.6,
                    label='GT Completed' if 'GT Completed' not in [h.get_label() for h in ax.get_legend_handles_labels()[0]] else "")

    # Plot Prediction Bars
    for action, timings in pred_timings.items():
        y_pos = action_y_map.get(action)
        if y_pos is None: continue

        # Plot Predicted 'in-progress' markers
        ax.plot(timings['in_progress'], [y_pos + 0.25] * len(timings['in_progress']), '|', markersize=15,
                color='darkorange', markeredgewidth=30,
                label='Predicted In Progress' if 'Predicted In Progress' not in [h.get_label() for h in ax.get_legend_handles_labels()[0]] else "")

        # Plot Predicted 'completed' markers
        ax.plot(timings['completed'], [y_pos - 0.25] * len(timings['completed']), '|', markersize=15,
                color='purple', markeredgewidth=30,
                label='Predicted Completed' if 'Predicted Completed' not in [h.get_label() for h in ax.get_legend_handles_labels()[0]] else "")


    # --- Chart Styling ---
    ax.set_yticks(np.arange(len(all_actions)))
    ax.set_yticklabels(all_actions, fontsize=10)
    ax.set_xlabel('Frame Number', fontsize=12)
    ax.set_ylabel('Actions', fontsize=12)
    ax.set_title('Comparison of Ground Truth vs. Predicted Action Timelines', fontsize=16, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.invert_yaxis()
    ax.set_xlim(0, max_frame + 50) # Add some padding

    # Create a clean legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower right', fontsize=10)

    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
    else:
        plt.show()

if __name__ == '__main__':
    # Ensure the JSON files are in the same directory or provide full paths.
    ground_truth_file = 'exp2_gt_new_condensed.json'
    prediction_file = 'RCWPS_Stack_gemini-2.5-flash-lite-preview-06-17_no_gaze_use_gt_result.json'

    visualize_comparison_timeline(ground_truth_file, prediction_file)