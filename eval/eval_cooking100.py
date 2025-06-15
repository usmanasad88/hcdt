import json
import argparse
import pandas as pd
from sklearn.metrics import classification_report
from collections import defaultdict

def load_json(file_path):
    """Loads a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        exit(1)

def load_dag(dag_path):
    """Loads the DAG and converts it to a dictionary for easy lookup."""
    dag_list = load_json(dag_path)
    return {step['id']: step for step in dag_list}

def normalize_value(value):
    """Converts string 'True'/'False' to boolean, and 'Unknown' to None."""
    if isinstance(value, str):
        if value.lower() == 'true':
            return True
        if value.lower() == 'false':
            return False
        if value.lower() == 'unknown':
            return None
    return value

def generate_ground_truth(total_frames, dag):
    """
    Generates a frame-by-frame ground truth based on interpretation of
    the keysteps and narration files. This is a 'best-effort' annotation.
    """
    # Initialize all frames with the starting state
    initial_state = {
        "pot_on_stove": False, "stove_burner_on": False, "water_in_pot": False,
        "water_boiling": False, "noodles_in_pot": False, "noodles_cooked": False,
        "cabbage_prepared": False, "garlic_prepared": False, "spring_onions_prepared": False,
        "oil_in_pan": False, "ingredients_in_pan_cooked": False, "noodles_drained": False,
        "food_on_plate": False, "steps_completed": set()
    }
    ground_truth = [initial_state.copy() for _ in range(total_frames + 1)]

    # Interpret keysteps and narration to update state over time
    # NOTE: Timestamps are converted to frame numbers (1s = 1 frame)
    
    # By 0:01, pot is on stove with water, burner is on.
    for i in range(1, total_frames + 1):
        ground_truth[i]["pot_on_stove"] = True
        ground_truth[i]["water_in_pot"] = True
        ground_truth[i]["stove_burner_on"] = True
        ground_truth[i]["water_boiling"] = True # Boiling starts quickly
        ground_truth[i]["steps_completed"].update(["step_1", "step_2"])

    # 0:01 - 0:04 Add noodles
    for i in range(4, total_frames + 1):
        ground_truth[i]["noodles_in_pot"] = True
        ground_truth[i]["steps_completed"].add("step_3")

    # 0:10 - 0:12 Place skillet on stove. Let's assume oil is added around the same time.
    # From narration 04:35, oil is added. So we'll use that instead.
    for i in range(10, total_frames + 1):
        ground_truth[i]["oil_in_pan"] = True
        ground_truth[i]["steps_completed"].add("step_8")
        
    # 01:21 - 01:31 Cut cabbage (step_5)
    for i in range(81, total_frames + 1): # 1*60 + 21 = 81
        ground_truth[i]["cabbage_prepared"] = True
        ground_truth[i]["steps_completed"].add("step_5")

    # The rest of the events like cutting garlic/onions happen after frame 100.
    # Noodles are not drained within the first 100 seconds.
    # Therefore, noodles_cooked and noodles_drained remain False.

    return ground_truth

def analyze_state_accuracy(predictions, ground_truth):
    """
    Calculates and prints the accuracy of state predictions.
    Ignores 'Unknown' predictions.
    """
    print("--- STATE VARIABLE ACCURACY ---")
    print("Metrics are calculated only on frames where the model provided a True/False prediction (ignoring 'Unknown').\n")

    report_data = {}
    state_keys = ground_truth[0].keys() - {"steps_completed"}

    for key in sorted(state_keys):
        y_true = []
        y_pred = []
        
        for i, pred_frame in enumerate(predictions):
            if i >= len(ground_truth): break
            
            predicted_value = normalize_value(pred_frame['state'].get(key))
            true_value = ground_truth[i].get(key)
            
            # Only score if the model made a concrete prediction
            if predicted_value is not None:
                y_true.append(true_value)
                y_pred.append(predicted_value)

        if not y_true:
            print(f"\nNo concrete predictions found for '{key}'. Skipping.")
            continue
        
        # Use classification_report to get precision, recall, f1-score
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        report_data[key] = {
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1-score': report['weighted avg']['f1-score'],
            'support': report['weighted avg']['support']
        }

    df = pd.DataFrame.from_dict(report_data, orient='index')
    df.index.name = 'State Variable'
    print(df.to_string())
    print("\n")


def analyze_dag_compliance(predictions, dag):
    """
    Checks for violations of the DAG's predecessor rules in the model's output.
    """
    print("--- DAG COMPLIANCE ANALYSIS ---")
    print("Checks if a completed step has all its required predecessors listed as complete.\n")
    
    violations = []
    total_completed_steps_checked = 0

    for frame in predictions:
        frame_num = frame['frame']
        completed_steps = set(frame['state'].get('steps_completed', []))
        
        if not completed_steps:
            continue

        total_completed_steps_checked += len(completed_steps)
        for step_id in completed_steps:
            step_info = dag.get(step_id)
            if not step_info:
                continue # Skip if step_id from model is not in our DAG
            
            predecessors = set(step_info['predecessors'])
            if not predecessors.issubset(completed_steps):
                missing = predecessors - completed_steps
                violations.append({
                    'frame': frame_num,
                    'step': step_id,
                    'missing_predecessors': list(missing)
                })
    
    violation_rate = (len(violations) / total_completed_steps_checked * 100) if total_completed_steps_checked > 0 else 0
    
    print(f"Total Frames Analyzed: {len(predictions)}")
    print(f"Total 'steps_completed' entries checked: {total_completed_steps_checked}")
    print(f"Total DAG Violations Found: {len(violations)}")
    print(f"Violation Rate: {violation_rate:.2f}%")

    if violations:
        print("\nFirst 5 Violations Found:")
        df_violations = pd.DataFrame(violations[:5])
        print(df_violations.to_string())
    print("\n")

def analyze_step_completion(predictions, ground_truth, dag):
    """
    Analyzes the accuracy of the 'steps_completed' list.
    """
    print("--- COMPLETED STEPS ACCURACY (Precision/Recall) ---")
    print("Evaluates the model's ability to correctly identify the set of completed steps at each frame.\n")
    
    # TP: True Positive, FP: False Positive, FN: False Negative
    step_metrics = {step_id: {'TP': 0, 'FP': 0, 'FN': 0} for step_id in dag.keys()}
    
    for i, pred_frame in enumerate(predictions):
        if i >= len(ground_truth): break
        
        pred_set = set(pred_frame['state'].get('steps_completed', []))
        gt_set = ground_truth[i].get('steps_completed', set())
        
        all_steps = pred_set.union(gt_set)
        
        for step_id in all_steps:
            if step_id not in step_metrics: continue
            
            in_pred = step_id in pred_set
            in_gt = step_id in gt_set
            
            if in_pred and in_gt:
                step_metrics[step_id]['TP'] += 1
            elif in_pred and not in_gt:
                step_metrics[step_id]['FP'] += 1
            elif not in_pred and in_gt:
                step_metrics[step_id]['FN'] += 1

    results = {}
    for step_id, counts in step_metrics.items():
        tp, fp, fn = counts['TP'], counts['FP'], counts['FN']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Only show steps that were relevant in the ground truth or prediction
        if (tp + fp + fn) > 0:
            results[step_id] = {
                'precision': precision,
                'recall': recall,
                'f1-score': f1_score,
                'support (frames)': tp + fn
            }
            
    df = pd.DataFrame.from_dict(results, orient='index')
    df.index.name = "Step ID"
    print(df.to_string())
    print("\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze AI model predictions for task completion.")
    parser.add_argument("results_file", nargs="?", default="data/Cooking/result_test.json", help="Path to the result_test.json file (default: data/Cooking/result_test.json).")
    parser.add_argument("dag_file", nargs="?", default="data/Cooking/dag_noodles_v2.json", help="Path to the dag_noodles_v2.json file (default: data/Cooking/dag_noodles_v2.json).")
    args = parser.parse_args()

    predictions = load_json(args.results_file)
    dag = load_dag(args.dag_file)
    
    # Generate ground truth based on analysis of narration/keysteps
    total_frames = len(predictions)
    ground_truth = generate_ground_truth(total_frames, dag)

    # --- Run Analyses ---
    analyze_state_accuracy(predictions, ground_truth)
    analyze_dag_compliance(predictions, dag)
    analyze_step_completion(predictions, ground_truth, dag)


if __name__ == "__main__":
    main()