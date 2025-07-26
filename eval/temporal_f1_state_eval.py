import json
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

def get_segments(data, key, is_continuous_state=False):
    """
    Extracts continuous temporal segments for actions from data.

    Args:
        data (list): The dataset (either GT or prediction).
        key (str): The key to look for ('steps_in_progress' or 'steps_completed').
        is_continuous_state (bool): If True, a segment ends only at the last frame
                                    once it has started (for 'steps_completed').

    Returns:
        dict: A dictionary where keys are action names and values are lists of
              segments, with each segment being a {'start': frame, 'end': frame} dict.
    """
    segments = defaultdict(list)
    active_segments = {} # {action: start_frame}
    max_frame = data[-1]['frame_number'] if data else 0

    for i, entry in enumerate(data):
        frame = entry['frame_number']
        actions_present = set(entry['state'].get(key, []))

        # Check for newly started actions
        for action in actions_present:
            if action not in active_segments:
                active_segments[action] = frame

        # Check for ended actions
        ended_actions = set()
        for action, start_frame in active_segments.items():
            if action not in actions_present:
                # The action is no longer in the list, so the segment ends.
                # The end frame is the current frame.
                segments[action].append({'start': start_frame, 'end': frame})
                ended_actions.add(action)

        for action in ended_actions:
            del active_segments[action]

    # Add any segments that are still active at the end of the video
    for action, start_frame in active_segments.items():
         segments[action].append({'start': start_frame, 'end': max_frame})

    # For continuous states like 'completed', once it starts it never ends.
    if is_continuous_state:
        for action, segs in segments.items():
            if segs:
                # Find the earliest start time and make it run to the end.
                min_start = min(s['start'] for s in segs)
                segments[action] = [{'start': min_start, 'end': max_frame}]

    return segments

def merge_prediction_markers(data, key, merge_gap=50):
    """Merges discrete prediction markers into continuous segments."""
    markers = defaultdict(list)
    for entry in data:
        frame = entry['frame_number']
        for action in entry['state'].get(key, []):
            markers[action].append(frame)

    segments = defaultdict(list)
    for action, frames in markers.items():
        if not frames:
            continue
        
        sorted_frames = sorted(frames)
        start = sorted_frames[0]
        end = sorted_frames[0]
        
        for i in range(1, len(sorted_frames)):
            if sorted_frames[i] - end <= merge_gap:
                end = sorted_frames[i]  # Extend the current segment
            else:
                segments[action].append({'start': start, 'end': end})
                start = sorted_frames[i] # Start a new segment
                end = sorted_frames[i]
        
        segments[action].append({'start': start, 'end': end}) # Add the last segment
        
    return segments


def calculate_segmental_f1(gt_data, pred_data, key, iou_threshold=0.5, merge_gap=15):
    """Calculates a segmental F1 score based on Intersection over Union (IoU)."""
    
    is_completed_key = (key == 'steps_completed')
    
    gt_segments = get_segments(gt_data, key, is_continuous_state=is_completed_key)
    pred_segments = merge_prediction_markers(pred_data, key, merge_gap=merge_gap)

    all_actions = set(gt_segments.keys()) | set(pred_segments.keys())
    
    total_tp, total_fp, total_fn = 0, 0, 0

    for action in all_actions:
        gt_segs_action = gt_segments.get(action, [])
        pred_segs_action = pred_segments.get(action, [])
        
        if not gt_segs_action and not pred_segs_action:
            continue

        tp = 0
        fp = 0
        
        matched_gt_indices = set()

        for pred_seg in pred_segs_action:
            best_iou = 0
            best_gt_idx = -1

            for i, gt_seg in enumerate(gt_segs_action):
                # Calculate Intersection
                intersection_start = max(pred_seg['start'], gt_seg['start'])
                intersection_end = min(pred_seg['end'], gt_seg['end'])
                intersection = max(0, intersection_end - intersection_start)
                
                # Calculate Union
                union_start = min(pred_seg['start'], gt_seg['start'])
                union_end = max(pred_seg['end'], gt_seg['end'])
                union = max(0, union_end - union_start)

                if union == 0: continue
                
                iou = intersection / union
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            if best_iou >= iou_threshold:
                if best_gt_idx not in matched_gt_indices:
                    tp += 1
                    matched_gt_indices.add(best_gt_idx)
                else:
                    # This prediction matched a GT segment that was already claimed
                    # by a better prediction. It's an FP.
                    fp += 1
            else:
                fp += 1
        
        fn = len(gt_segs_action) - len(matched_gt_indices)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1, precision, recall

def calculate_framewise_f1(gt_data, pred_data):
    """Calculates a direct, frame-by-frame F1 score for 'steps_available'."""
    
    # Create a map of GT frames for easy lookup
    gt_map = {entry['frame_number']: entry['state'].get('steps_available', []) for entry in gt_data}
    gt_frames = sorted(gt_map.keys())

    total_tp, total_fp, total_fn = 0, 0, 0
    
    for pred_entry in pred_data:
        pred_frame = pred_entry['frame_number']
        pred_set = set(pred_entry['state'].get('steps_available', []))
        
        # Find the closest preceding ground truth frame
        # This handles cases where prediction frames don't perfectly align with GT frames
        closest_gt_frame = max([f for f in gt_frames if f <= pred_frame], default=None)
        if closest_gt_frame is None:
            continue
            
        gt_set = set(gt_map[closest_gt_frame])
        
        tp = len(pred_set.intersection(gt_set))
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1, precision, recall


def main():
    """Main function to load data, calculate scores, and print results."""
    ground_truth_file = 'exp2_gt_new_condensed.json'
    prediction_file = 'RCWPS_Stack_gemini-2.5-flash-lite-preview-06-17_use_gaze_use_gt_result.json'
    
    gt_data = load_json_data(ground_truth_file)
    pred_data = load_json_data(prediction_file)
    
    if gt_data is None or pred_data is None:
        return

    # --- Configuration ---
    IOU_THRESHOLD = 0.5
    MERGE_GAP_FRAMES = 15 # Frames within this gap will be merged into one segment
    
    # --- Calculate Scores ---
    f1_completed, _, _ = calculate_segmental_f1(
        gt_data, pred_data, 'steps_completed', IOU_THRESHOLD, MERGE_GAP_FRAMES
    )
    
    f1_in_progress, _, _ = calculate_segmental_f1(
        gt_data, pred_data, 'steps_in_progress', IOU_THRESHOLD, MERGE_GAP_FRAMES
    )
    
    f1_available, _, _ = calculate_framewise_f1(gt_data, pred_data)
    
    # --- Calculate Weighted Score ---
    weights = {'completed': 0.5, 'in_progress': 0.3, 'available': 0.2}
    
    combined_f1 = (weights['completed'] * f1_completed +
                   weights['in_progress'] * f1_in_progress +
                   weights['available'] * f1_available)

    # --- Print Results ---
    print("="*50)
    print("      Model Performance Evaluation Report")
    print("="*50)
    print(f"Parameters:")
    print(f"  - IoU Threshold for Segments: {IOU_THRESHOLD}")
    print(f"  - Prediction Merge Gap:       {MERGE_GAP_FRAMES} frames\n")
    
    print("--- Individual F1 Scores ---")
    print(f"  Steps Completed   (Weight: {weights['completed']*100}%): {f1_completed:.4f}")
    print(f"  Steps In Progress (Weight: {weights['in_progress']*100}%): {f1_in_progress:.4f}")
    print(f"  Steps Available   (Weight: {weights['available']*100}%): {f1_available:.4f}\n")
    
    print("--- Final Combined Score ---")
    print(f"Weighted F1 Score: {combined_f1:.4f}")
    print("="*50)


if __name__ == '__main__':
    main()
