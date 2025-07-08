import json
import argparse
from typing import List, Dict, Any, Set, Optional

def check_frame_coverage(json_file_path: str, frame_step: int, max_frame_number: int) -> bool:
    """
    Checks if a JSON file contains results for all expected frames.
    The expected frames are generated starting from 1, incrementing by frame_step,
    up to max_frame_number.

    Args:
        json_file_path (str): Path to the input JSON file.
        frame_step (int): The step between frames to check for.
        max_frame_number (int): The maximum frame number to check up to.

    Returns:
        bool: True if all expected frames are present in the JSON file, False otherwise.
              Prints missing frames if any are found.
    """
    if frame_step <= 0:
        print(f"Error: frame_step must be a positive integer. Received: {frame_step}")
        return False
    if max_frame_number < 1:
        print(f"Error: max_frame_number must be at least 1. Received: {max_frame_number}")
        return False

    try:
        with open(json_file_path, 'r') as f:
            data: List[Dict[str, Any]] = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at '{json_file_path}'")
        return False
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{json_file_path}'")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while reading the file '{json_file_path}': {e}")
        return False

    if not isinstance(data, list):
        print(f"Error: JSON content in '{json_file_path}' is not a list.")
        return False

    existing_frames_in_file: Set[int] = set()
    for item_idx, item in enumerate(data):
        if isinstance(item, dict) and "frame" in item:
            frame_val = item["frame"]
            if isinstance(frame_val, int):
                existing_frames_in_file.add(frame_val)
            else:
                print(f"Warning: Item at index {item_idx} has a non-integer 'frame' value: '{frame_val}'. Skipping this frame value.")
        else:
            print(f"Warning: Item at index {item_idx} is not a dictionary or is missing the 'frame' key: '{str(item)[:100]}...'. Skipping this item.")
    
    # Generate the set of expected frame numbers
    # The sequence starts from 1 and goes up to max_frame_number with the given step.
    expected_frames: Set[int] = set(range(1, max_frame_number + 1, frame_step))

    if not expected_frames:
        print(f"Warning: No frames are expected with max_frame_number={max_frame_number} and frame_step={frame_step} starting from 1.")
        # If no frames are expected, and the file is empty or contains no relevant frames, it could be considered covered.
        # However, if the file contains frames, it's an odd situation.
        # For now, if no frames are expected, we'll say coverage is met.
        return True


    missing_frames = expected_frames - existing_frames_in_file

    if not missing_frames:
        print(f"All {len(expected_frames)} expected frames (up to {max_frame_number}, step {frame_step}) are present in '{json_file_path}'.")
        return True
    else:
        print(f"Missing {len(missing_frames)} out of {len(expected_frames)} expected frames in '{json_file_path}' (up to {max_frame_number}, step {frame_step}):")
        for f_num in sorted(list(missing_frames)):
            print(f"  - Frame {f_num}")
        return False



def calculate_set_metrics(prediction: List[str], ground_truth: List[str]) -> Dict[str, float]:
    """
    Calculates Precision, Recall, F1-Score, and counts (TP, FP, FN)
    for two lists by treating them as sets.

    Returns:
        A dictionary containing precision, recall, f1-score, tp, fp, and fn.
    """
    pred_set = set(prediction)
    gt_set = set(ground_truth)

    true_positives = len(pred_set.intersection(gt_set))
    false_positives = len(pred_set.difference(gt_set))
    false_negatives = len(gt_set.difference(pred_set))

    # Calculate Precision
    if true_positives + false_positives == 0:
        precision = 1.0 if not gt_set else 0.0
    else:
        precision = true_positives / (true_positives + false_positives)

    # Calculate Recall
    if true_positives + false_negatives == 0:
        recall = 1.0 if not pred_set else 0.0
    else:
        recall = true_positives / (true_positives + false_negatives)

    # Calculate F1-Score
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
    }

def evaluate_frames(gt_data: List[Dict[str, Any]], pred_data: List[Dict[str, Any]]) -> Dict:
    """
    Compares ground truth and prediction data frame by frame and computes metrics.
    Handles different data structures for Stack dataset.

    Returns:
        A dictionary containing per-frame metrics and aggregate metrics.
    """
    # Handle different data structures
    # GT data: direct state data with frame_number field
    # Pred data: nested structure with frame and state fields
    gt_states = {}
    pred_states = {}
    
    # Process ground truth data (Stack format)
    for item in gt_data:
        if 'frame_number' in item:
            frame_num = item['frame_number']
            # The entire item is the state for Stack dataset
            gt_states[frame_num] = item
        elif 'frame' in item and 'state' in item:
            # HAViD format fallback
            gt_states[item['frame']] = item['state']
    
    # Process prediction data
    for item in pred_data:
        if 'frame' in item and 'state' in item:
            pred_states[item['frame']] = item['state']

    per_frame_metrics = []
    
    sorted_frames = sorted(gt_states.keys())
    for frame in sorted_frames:
        if frame not in pred_states:
            print(f"Warning: Frame {frame} not found in prediction file. Skipping.")
            continue

        gt_state = gt_states[frame]
        pred_state = pred_states[frame]
        
        frame_metrics = {'frame': frame}

        # 1. Accuracy for categorical fields
        frame_metrics['holding_acc'] = 1.0 if pred_state['operator_holding'] == gt_state['operator_holding'] else 0.0
        frame_metrics['gaze_acc'] = 1.0 if pred_state['gaze_target'] == gt_state['gaze_target'] else 0.0
        
        # 2. Accuracy for boolean fields (Stack-specific)
        if 'toolbox_placed_on_table' in gt_state and 'toolbox_placed_on_table' in pred_state:
            frame_metrics['toolbox_acc'] = 1.0 if pred_state['toolbox_placed_on_table'] == gt_state['toolbox_placed_on_table'] else 0.0
        
        # 3. Accuracy for numeric fields (Stack-specific)
        if 'num_chairs_stacked' in gt_state and 'num_chairs_stacked' in pred_state:
            frame_metrics['num_chairs_acc'] = 1.0 if pred_state['num_chairs_stacked'] == gt_state['num_chairs_stacked'] else 0.0
        
        # 4. Metrics for list fields (handle both string and list formats)
        def safe_get_list(state, key):
            """Convert steps_in_progress to list if it's a string"""
            value = state.get(key, [])
            if isinstance(value, str):
                return [value] if value != "idle" else []
            return value if isinstance(value, list) else []
        
        gt_completed = safe_get_list(gt_state, 'steps_completed')
        pred_completed = safe_get_list(pred_state, 'steps_completed')
        gt_inprogress = safe_get_list(gt_state, 'steps_in_progress')
        pred_inprogress = safe_get_list(pred_state, 'steps_in_progress')
        gt_available = safe_get_list(gt_state, 'steps_available')
        pred_available = safe_get_list(pred_state, 'steps_available')
        
        frame_metrics['completed_metrics'] = calculate_set_metrics(pred_completed, gt_completed)
        frame_metrics['inprogress_metrics'] = calculate_set_metrics(pred_inprogress, gt_inprogress)
        frame_metrics['available_metrics'] = calculate_set_metrics(pred_available, gt_available)
        
        # 5. List comparison metrics (Stack-specific)
        if 'list_of_stacked_chairs' in gt_state and 'list_of_stacked_chairs' in pred_state:
            frame_metrics['stacked_chairs_metrics'] = calculate_set_metrics(
                pred_state['list_of_stacked_chairs'], 
                gt_state['list_of_stacked_chairs']
            )
        
        per_frame_metrics.append(frame_metrics)

    if not per_frame_metrics:
        return {"aggregate": {}, "per_frame": []}

    num_frames = len(per_frame_metrics)
    
    # 3. Aggregate metrics
    aggregate_metrics = {
        "mean_holding_accuracy": sum(m['holding_acc'] for m in per_frame_metrics) / num_frames,
        "mean_gaze_accuracy": sum(m['gaze_acc'] for m in per_frame_metrics) / num_frames,
        "mean_completed_f1": sum(m['completed_metrics']['f1'] for m in per_frame_metrics) / num_frames,
        "mean_inprogress_f1": sum(m['inprogress_metrics']['f1'] for m in per_frame_metrics) / num_frames,
        "mean_available_f1": sum(m['available_metrics']['f1'] for m in per_frame_metrics) / num_frames,
    }
    
    # Add Stack-specific metrics if available
    if any('toolbox_acc' in m for m in per_frame_metrics):
        aggregate_metrics["mean_toolbox_accuracy"] = sum(m.get('toolbox_acc', 0) for m in per_frame_metrics) / num_frames
    
    if any('num_chairs_acc' in m for m in per_frame_metrics):
        aggregate_metrics["mean_num_chairs_accuracy"] = sum(m.get('num_chairs_acc', 0) for m in per_frame_metrics) / num_frames
    
    if any('stacked_chairs_metrics' in m for m in per_frame_metrics):
        aggregate_metrics["mean_stacked_chairs_f1"] = sum(m.get('stacked_chairs_metrics', {}).get('f1', 0) for m in per_frame_metrics) / num_frames
    
    aggregate_metrics['overall_score'] = sum(aggregate_metrics.values()) / len(aggregate_metrics)
    
    # 4. Aggregate counts
    aggregate_counts = {
        "completed": {
            "tp": sum(m['completed_metrics']['tp'] for m in per_frame_metrics),
            "fp": sum(m['completed_metrics']['fp'] for m in per_frame_metrics),
            "fn": sum(m['completed_metrics']['fn'] for m in per_frame_metrics),
        },
        "inprogress": {
            "tp": sum(m['inprogress_metrics']['tp'] for m in per_frame_metrics),
            "fp": sum(m['inprogress_metrics']['fp'] for m in per_frame_metrics),
            "fn": sum(m['inprogress_metrics']['fn'] for m in per_frame_metrics),
        },
        "available": {
            "tp": sum(m['available_metrics']['tp'] for m in per_frame_metrics),
            "fp": sum(m['available_metrics']['fp'] for m in per_frame_metrics),
            "fn": sum(m['available_metrics']['fn'] for m in per_frame_metrics),
        },
    }
    
    # Add Stack-specific counts if available
    if any('stacked_chairs_metrics' in m for m in per_frame_metrics):
        aggregate_counts["stacked_chairs"] = {
            "tp": sum(m.get('stacked_chairs_metrics', {}).get('tp', 0) for m in per_frame_metrics),
            "fp": sum(m.get('stacked_chairs_metrics', {}).get('fp', 0) for m in per_frame_metrics),
            "fn": sum(m.get('stacked_chairs_metrics', {}).get('fn', 0) for m in per_frame_metrics),
        }
    
    return {
        "aggregate_metrics": aggregate_metrics,
        "aggregate_counts": aggregate_counts,
        "per_frame": per_frame_metrics
    }

def analyze_step_completion_timing(
    gt_data: List[Dict[str, Any]], 
    pred_data: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Optional[int]]]:
    """
    Analyzes and compares the frame numbers at which steps are first marked as completed
    in the ground truth versus the prediction data.
    Updated to handle Stack dataset structure.

    Args:
        gt_data: List of ground truth frame states.
        pred_data: List of prediction frame states.

    Returns:
        A dictionary where keys are step names. Each value is another dictionary
        with 'gt_completion_frame' and 'pred_first_completion_frame'.
        Values can be None if a step is never marked completed.
    """
    gt_first_completion: Dict[str, int] = {}
    pred_first_completion: Dict[str, int] = {}
    all_steps: Set[str] = set()

    def safe_get_list(state, key):
        """Convert steps to list if it's a string"""
        value = state.get(key, [])
        if isinstance(value, str):
            return [value] if value != "idle" else []
        return value if isinstance(value, list) else []

    # Process Ground Truth data
    for item in gt_data:
        # Handle Stack format (frame_number) vs HAViD format (frame + state)
        if 'frame_number' in item:
            frame_num = item['frame_number']
            state = item  # The entire item is the state for Stack dataset
        elif 'frame' in item and 'state' in item:
            frame_num = item['frame']
            state = item['state']
        else:
            continue
        
        completed_steps = safe_get_list(state, 'steps_completed')
        for step in completed_steps:
            all_steps.add(step)
            if step not in gt_first_completion:
                gt_first_completion[step] = frame_num
        
        # Also collect steps from in_progress and available to have a full list of steps
        for step in safe_get_list(state, 'steps_in_progress'):
            all_steps.add(step)
        for step in safe_get_list(state, 'steps_available'):
            all_steps.add(step)

    # Process Prediction data
    for item in pred_data:
        if 'frame' in item and 'state' in item:
            frame_num = item['frame']
            state = item['state']
        else:
            continue

        completed_steps = safe_get_list(state, 'steps_completed')
        for step in completed_steps:
            all_steps.add(step)
            if step not in pred_first_completion:
                pred_first_completion[step] = frame_num
        
        # Also collect steps from in_progress and available from predictions
        for step in safe_get_list(state, 'steps_in_progress'):
            all_steps.add(step)
        for step in safe_get_list(state, 'steps_available'):
            all_steps.add(step)

    # Compile results
    completion_timing_analysis: Dict[str, Dict[str, Optional[int]]] = {}
    for step in sorted(list(all_steps)): # Iterate in a consistent order
        completion_timing_analysis[step] = {
            "gt_completion_frame": gt_first_completion.get(step),
            "pred_first_completion_frame": pred_first_completion.get(step)
        }
        
    return completion_timing_analysis

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a model's state prediction against ground truth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )    
    # parser.add_argument("ground_truth_file", nargs="?", default="data/HAViD/S02A08I21_gt.json", help="Path to the ground truth JSON file. Defaults to '/home/mani/Repos/hcdt/data/HAViD/S13A11I21_gt.json'.")
    parser.add_argument("ground_truth_file", nargs="?", default="data/Stack/cam1_gt.json", help="Path to the ground truth JSON file. Defaults to '/home/mani/Repos/hcdt/data/HAViD/S13A11I21_gt.json'.")
    # parser.add_argument("prediction_file", nargs="?", default="logs/ICL_result_havid_ex0002.json", help="Path to the model's prediction JSON file.")
    parser.add_argument("prediction_file", nargs="?", default="data/Stack/ICL_result.json", help="Path to the model's prediction JSON file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print metrics for each individual frame.")
    parser.add_argument("--analyze-timing", action="store_true", help="Analyze and print step completion timing.")

    args = parser.parse_args()

    try:
        with open(args.ground_truth_file, 'r') as f:
            gt_data = json.load(f)
        with open(args.prediction_file, 'r') as f:
            pred_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return

    results = evaluate_frames(gt_data, pred_data)
    
    if args.verbose and results["per_frame"]:
        print("--- PER-FRAME METRICS & COUNTS ---")
        for fm in results["per_frame"]:
            print(f"\nFrame {fm['frame']}:")
            print(f"  - Accuracy: Holding={fm['holding_acc']:.2f}, Gaze={fm['gaze_acc']:.2f}", end="")
            
            # Add Stack-specific accuracies if available
            if 'toolbox_acc' in fm:
                print(f", Toolbox={fm['toolbox_acc']:.2f}", end="")
            if 'num_chairs_acc' in fm:
                print(f", NumChairs={fm['num_chairs_acc']:.2f}", end="")
            print()  # New line
            
            cm = fm['completed_metrics']
            im = fm['inprogress_metrics']
            am = fm['available_metrics']
            
            print(f"  - Completed:   F1={cm['f1']:.2f} (TP:{cm['tp']}, FP:{cm['fp']}, FN:{cm['fn']})")
            print(f"  - In-Progress: F1={im['f1']:.2f} (TP:{im['tp']}, FP:{im['fp']}, FN:{im['fn']})")
            print(f"  - Available:   F1={am['f1']:.2f} (TP:{am['tp']}, FP:{am['fp']}, FN:{am['fn']})")
            
            # Add stacked chairs metrics if available
            if 'stacked_chairs_metrics' in fm:
                scm = fm['stacked_chairs_metrics']
                print(f"  - Stacked Chairs: F1={scm['f1']:.2f} (TP:{scm['tp']}, FP:{scm['fp']}, FN:{scm['fn']})")
        print("-" * 35)

    if results["aggregate_metrics"]:
        agg_met = results["aggregate_metrics"]
        agg_cnt = results["aggregate_counts"]
        
        print("\n--- AGGREGATE PERFORMANCE METRICS ---")
        print(f"Total frames evaluated: {len(results['per_frame'])}")
        print(f"  Mean Holding Accuracy:      {agg_met['mean_holding_accuracy']:.4f}")
        print(f"  Mean Gaze Accuracy:         {agg_met['mean_gaze_accuracy']:.4f}")
        print(f"  Mean F1 (Steps Completed):  {agg_met['mean_completed_f1']:.4f}")
        print(f"  Mean F1 (Steps In Progress):{agg_met['mean_inprogress_f1']:.4f}")
        print(f"  Mean F1 (Steps Available):  {agg_met['mean_available_f1']:.4f}")
        
        # Add Stack-specific metrics if available
        if 'mean_toolbox_accuracy' in agg_met:
            print(f"  Mean Toolbox Accuracy:      {agg_met['mean_toolbox_accuracy']:.4f}")
        if 'mean_num_chairs_accuracy' in agg_met:
            print(f"  Mean Num Chairs Accuracy:   {agg_met['mean_num_chairs_accuracy']:.4f}")
        if 'mean_stacked_chairs_f1' in agg_met:
            print(f"  Mean F1 (Stacked Chairs):   {agg_met['mean_stacked_chairs_f1']:.4f}")
        
        print("---------------------------------------")
        print(f"  OVERALL MODEL SCORE:        {agg_met['overall_score']:.4f}")
        
        print("\n--- AGGREGATE COUNTS (Total TP/FP/FN across all frames) ---")
        print(f"  Steps Completed:    TP: {agg_cnt['completed']['tp']}, FP: {agg_cnt['completed']['fp']}, FN: {agg_cnt['completed']['fn']}")
        print(f"  Steps In Progress:  TP: {agg_cnt['inprogress']['tp']}, FP: {agg_cnt['inprogress']['fp']}, FN: {agg_cnt['inprogress']['fn']}")
        print(f"  Steps Available:    TP: {agg_cnt['available']['tp']}, FP: {agg_cnt['available']['fp']}, FN: {agg_cnt['available']['fn']}")
        
        # Add stacked chairs counts if available
        if 'stacked_chairs' in agg_cnt:
            print(f"  Stacked Chairs:     TP: {agg_cnt['stacked_chairs']['tp']}, FP: {agg_cnt['stacked_chairs']['fp']}, FN: {agg_cnt['stacked_chairs']['fn']}")

    else:
        print("No matching frames were found to evaluate.")
    
    if not args.analyze_timing:
        print("\n--- STEP COMPLETION TIMING ANALYSIS ---")
        timing_analysis = analyze_step_completion_timing(gt_data, pred_data)
        if not timing_analysis:
            print("No steps found to analyze for timing.")
        else:
            print(f"{'Step':<30} | {'GT Completion Frame':<20} | {'Pred First Completion Frame':<25} | {'Difference':<10}")
            print("-" * 95)
            for step, times in timing_analysis.items():
                gt_frame = times['gt_completion_frame']
                pred_frame = times['pred_first_completion_frame']
                
                gt_str = str(gt_frame) if gt_frame is not None else "N/A (Not Completed)"
                pred_str = str(pred_frame) if pred_frame is not None else "N/A (Not Completed)"
                
                diff_str = "N/A"
                if gt_frame is not None and pred_frame is not None:
                    difference = pred_frame - gt_frame
                    diff_str = f"{difference}"
                    if difference > 0:
                        diff_str += " (Late)"
                    elif difference < 0:
                        diff_str += " (Early)"
                    else:
                        diff_str += " (On Time)"
                elif gt_frame is not None and pred_frame is None:
                    diff_str = "Missed"
                elif gt_frame is None and pred_frame is not None:
                    diff_str = "False Positive Completion"


                print(f"{step:<30} | {gt_str:<20} | {pred_str:<25} | {diff_str:<10}")



if __name__ == "__main__":
    # check_frame_coverage("/home/mani/Repos/hcdt/data/HAViD/S13A11I21_gt.json", 15, 1786)
    main()
    