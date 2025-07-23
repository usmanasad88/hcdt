"""
Evaluation script to compare ICL results against ground truth.
Includes F1 scores for steps, BERT scores for natural language, and temporal analysis.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import defaultdict
import sys
import os

# Add the parent directory to the Python path to import from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.textutils import get_ground_truth

# Try to import BERT score - if not available, use a simple string similarity
try:
    from bert_score import score as bert_score
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("Warning: bert_score not available. Using Jaccard similarity for text comparison.")

def jaccard_similarity(text1: str, text2: str) -> float:
    """Simple Jaccard similarity for text comparison when BERT is not available."""
    if not text1 or not text2:
        return 0.0
    
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0

def compute_text_similarity(pred_texts: List[str], gt_texts: List[str]) -> float:
    """Compute similarity between predicted and ground truth texts."""
    global BERT_AVAILABLE
    
    if not pred_texts or not gt_texts:
        return 0.0
    
    if BERT_AVAILABLE:
        # Use BERT score
        try:
            P, R, F1 = bert_score(pred_texts, gt_texts, lang="en", verbose=False)
            return F1.mean().item()
        except Exception as e:
            print(f"BERT score failed: {e}. Falling back to Jaccard similarity.")
            BERT_AVAILABLE = False
    
    if not BERT_AVAILABLE:
        # Use Jaccard similarity
        similarities = []
        for pred, gt in zip(pred_texts, gt_texts):
            if pred is None:
                pred = ""
            if gt is None:
                gt = ""
            similarities.append(jaccard_similarity(str(pred), str(gt)))
        return np.mean(similarities) if similarities else 0.0

def extract_boolean_states(state: Dict) -> Dict[str, bool]:
    """Extract boolean state variables from a state dictionary."""
    # Generic boolean extraction - find all boolean values in the state
    boolean_states = {}
    for key, value in state.items():
        if isinstance(value, bool):
            boolean_states[key] = value
        # Also check for common boolean-like keys that might be stored as other types
        elif key in ["toolbox_placed_on_table"] and value is not None:
            boolean_states[key] = bool(value)
    
    return boolean_states

def normalize_step_list(steps: List[str]) -> Set[str]:
    """Normalize step lists for comparison by converting to lowercase and stripping."""
    if not steps:
        return set()
    return {step.lower().strip() for step in steps if step}

def compute_step_f1(pred_steps: List[str], gt_steps: List[str]) -> Dict[str, float]:
    """Compute F1, precision, and recall for step lists."""
    pred_set = normalize_step_list(pred_steps)
    gt_set = normalize_step_list(gt_steps)
    
    if not pred_set and not gt_set:
        return {"f1": 1.0, "precision": 1.0, "recall": 1.0}
    
    if not pred_set:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    
    if not gt_set:
        return {"f1": 0.0, "precision": 0.0, "recall": 1.0}
    
    intersection = pred_set.intersection(gt_set)
    
    precision = len(intersection) / len(pred_set) if pred_set else 0.0
    recall = len(intersection) / len(gt_set) if gt_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {"f1": f1, "precision": precision, "recall": recall}

def compute_union_step_f1(pred_in_progress: List[str], pred_available: List[str], 
                         gt_in_progress: List[str], gt_available: List[str]) -> Dict[str, float]:
    """Compute F1, precision, and recall for the union of in_progress and available steps."""
    # Create union sets
    pred_union = normalize_step_list(pred_in_progress).union(normalize_step_list(pred_available))
    gt_union = normalize_step_list(gt_in_progress).union(normalize_step_list(gt_available))
    
    if not pred_union and not gt_union:
        return {"f1": 1.0, "precision": 1.0, "recall": 1.0}
    
    if not pred_union:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    
    if not gt_union:
        return {"f1": 0.0, "precision": 0.0, "recall": 1.0}
    
    intersection = pred_union.intersection(gt_union)
    
    precision = len(intersection) / len(pred_union) if pred_union else 0.0
    recall = len(intersection) / len(gt_union) if gt_union else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {"f1": f1, "precision": precision, "recall": recall}

def find_step_completion_timing(icl_results: List[Dict], gt_filename: str) -> Dict[str, Dict]:
    """
    Analyze when steps were first predicted as completed vs when they actually completed.
    Returns timing analysis for each step.
    """
    # Track when each step was first predicted as completed
    predicted_completions = {}
    
    # Track when each step was actually completed in ground truth
    actual_completions = {}
    
    # Process ICL results to find first prediction of completion
    for entry in icl_results:
        frame = entry["frame"]
        completed_steps = entry["state"].get("steps_completed", [])
        
        for step in completed_steps:
            step_norm = step.lower().strip()
            if step_norm not in predicted_completions:
                predicted_completions[step_norm] = frame
    
    # Process ground truth to find actual completion times
    # Load ground truth data
    with open(gt_filename, 'r') as f:
        gt_data = json.load(f)
    
    # Track previous completed steps to detect new completions
    prev_completed = set()
    
    for entry in gt_data:
        frame = entry.get("frame_number", entry.get("frame", 0))
        state = entry.get("state", {})
        completed_steps = state.get("steps_completed", [])
        
        current_completed = normalize_step_list(completed_steps)
        newly_completed = current_completed - prev_completed
        
        for step in newly_completed:
            if step not in actual_completions:
                actual_completions[step] = frame
        
        prev_completed = current_completed
    
    # Compute timing differences
    timing_analysis = {}
    
    for step in set(predicted_completions.keys()).union(set(actual_completions.keys())):
        pred_frame = predicted_completions.get(step)
        actual_frame = actual_completions.get(step)
        
        analysis = {
            "predicted_frame": pred_frame,
            "actual_frame": actual_frame,
            "difference": None,
            "early": None,
            "late": None,
            "correct": None
        }
        
        if pred_frame is not None and actual_frame is not None:
            diff = pred_frame - actual_frame
            analysis["difference"] = diff
            analysis["early"] = diff < 0
            analysis["late"] = diff > 0
            analysis["correct"] = diff == 0
        
        timing_analysis[step] = analysis
    
    return timing_analysis

def evaluate_icl_results(icl_filename: str, gt_filename: str) -> Dict:
    """
    Comprehensive evaluation of ICL results against ground truth.
    """
    print(f"Loading ICL results from: {icl_filename}")
    print(f"Loading ground truth from: {gt_filename}")
    
    # Load ICL results
    with open(icl_filename, 'r') as f:
        icl_results = json.load(f)
    
    # Initialize metrics storage
    boolean_states_pred = []
    boolean_states_gt = []
    
    steps_completed_metrics = []
    steps_in_progress_metrics = []
    steps_available_metrics = []
    steps_available_or_in_progress_metrics = []
    
    current_keystep_pred = []
    current_keystep_gt = []
    next_keystep_pred = []
    next_keystep_gt = []
    
    operator_holding_pred = []
    operator_holding_gt = []
    gaze_target_pred = []
    gaze_target_gt = []
    
    # Process each frame
    for entry in icl_results:
        frame = entry["frame"]
        pred_state = entry["state"]
        
        # Get corresponding ground truth
        gt_state = get_ground_truth(frame, gt_filename)
        
        if gt_state is None:
            print(f"Warning: No ground truth found for frame {frame}")
            continue
        
        # Extract boolean states (keep for backwards compatibility but won't use in overall score)
        pred_bools = extract_boolean_states(pred_state)
        gt_bools = extract_boolean_states(gt_state)
        
        # Align boolean states (only include keys present in both)
        common_keys = set(pred_bools.keys()).intersection(set(gt_bools.keys()))
        
        for key in common_keys:
            boolean_states_pred.append(pred_bools[key])
            boolean_states_gt.append(gt_bools[key])
        
        # Evaluate steps
        steps_completed_metrics.append(compute_step_f1(
            pred_state.get("steps_completed", []),
            gt_state.get("steps_completed", [])
        ))
        
        steps_in_progress_metrics.append(compute_step_f1(
            pred_state.get("steps_in_progress", []),
            gt_state.get("steps_in_progress", [])
        ))
        
        steps_available_metrics.append(compute_step_f1(
            pred_state.get("steps_available", []),
            gt_state.get("steps_available", [])
        ))
        
        # Evaluate union of steps_in_progress and steps_available
        steps_available_or_in_progress_metrics.append(compute_union_step_f1(
            pred_state.get("steps_in_progress", []),
            pred_state.get("steps_available", []),
            gt_state.get("steps_in_progress", []),
            gt_state.get("steps_available", [])
        ))
        
        # Collect text fields for similarity computation
        current_keystep_pred.append(pred_state.get("current_keystep"))
        current_keystep_gt.append(gt_state.get("current_keystep"))
        
        next_keystep_pred.append(pred_state.get("next_keystep"))
        next_keystep_gt.append(gt_state.get("next_keystep"))
        
        operator_holding_pred.append(pred_state.get("operator_holding"))
        operator_holding_gt.append(gt_state.get("operator_holding"))
        
        gaze_target_pred.append(pred_state.get("gaze_target"))
        gaze_target_gt.append(gt_state.get("gaze_target"))
    
    # Compute overall metrics
    results = {}
    
    # Boolean states F1 score
    if boolean_states_pred and boolean_states_gt:
        results["boolean_states"] = {
            "f1": f1_score(boolean_states_gt, boolean_states_pred, average='weighted'),
            "precision": precision_score(boolean_states_gt, boolean_states_pred, average='weighted'),
            "recall": recall_score(boolean_states_gt, boolean_states_pred, average='weighted'),
            "total_comparisons": len(boolean_states_pred)
        }
    
    # Steps metrics
    def aggregate_step_metrics(metrics_list):
        if not metrics_list:
            return {"f1": 0.0, "precision": 0.0, "recall": 0.0}
        
        return {
            "f1": np.mean([m["f1"] for m in metrics_list]),
            "precision": np.mean([m["precision"] for m in metrics_list]),
            "recall": np.mean([m["recall"] for m in metrics_list])
        }
    
    results["steps_completed"] = aggregate_step_metrics(steps_completed_metrics)
    results["steps_in_progress"] = aggregate_step_metrics(steps_in_progress_metrics)
    results["steps_available"] = aggregate_step_metrics(steps_available_metrics)
    results["steps_available_or_in_progress"] = aggregate_step_metrics(steps_available_or_in_progress_metrics)
    
    # Text similarity metrics
    results["current_keystep_similarity"] = compute_text_similarity(
        [str(x) if x is not None else "" for x in current_keystep_pred],
        [str(x) if x is not None else "" for x in current_keystep_gt]
    )
    
    results["next_keystep_similarity"] = compute_text_similarity(
        [str(x) if x is not None else "" for x in next_keystep_pred],
        [str(x) if x is not None else "" for x in next_keystep_gt]
    )
    
    results["operator_holding_similarity"] = compute_text_similarity(
        [str(x) if x is not None else "" for x in operator_holding_pred],
        [str(x) if x is not None else "" for x in operator_holding_gt]
    )
    
    results["gaze_target_similarity"] = compute_text_similarity(
        [str(x) if x is not None else "" for x in gaze_target_pred],
        [str(x) if x is not None else "" for x in gaze_target_gt]
    )
    
    # Timing analysis
    print("Computing step completion timing analysis...")
    timing_analysis = find_step_completion_timing(icl_results, gt_filename)
    results["timing_analysis"] = timing_analysis
    
    # Compute timing statistics
    timing_diffs = []
    early_count = 0
    late_count = 0
    correct_count = 0
    
    for step, analysis in timing_analysis.items():
        if analysis["difference"] is not None:
            timing_diffs.append(analysis["difference"])
            if analysis["early"]:
                early_count += 1
            elif analysis["late"]:
                late_count += 1
            else:
                correct_count += 1
    
    if timing_diffs:
        results["timing_statistics"] = {
            "mean_difference": np.mean(timing_diffs),
            "std_difference": np.std(timing_diffs),
            "median_difference": np.median(timing_diffs),
            "early_predictions": early_count,
            "late_predictions": late_count,
            "correct_predictions": correct_count,
            "total_compared": len(timing_diffs)
        }
    
    # Compute overall score (weighted average) - Updated for universal metric across all experiments
    weights = {
        "steps_completed": 0.5,
        "steps_in_progress": 0.10,
        "steps_available": 0.15,
        "steps_available_or_in_progress": 0.2,
        "operator_holding_similarity": 0.025,
        "gaze_target_similarity": 0.025
    }
    
    overall_score = 0.0
    for metric, weight in weights.items():
        if metric in results:
            if metric.endswith("_similarity"):
                score = results[metric]
            else:
                score = results[metric].get("f1", 0.0)
            overall_score += weight * score
    
    results["overall_score"] = overall_score
    
    return results

def print_evaluation_results(results: Dict):
    """Print comprehensive evaluation results."""
    print("\n" + "="*80)
    print("ICL EVALUATION RESULTS")
    print("="*80)
    
    print(f"\nOVERALL SCORE: {results.get('overall_score', 0.0):.4f}")
    
    # Show the weights used in overall score calculation
    print("\n" + "-"*50)
    print("OVERALL SCORE COMPOSITION (Universal Across All Experiments)")
    print("-"*50)
    print("Metric                          Weight    Score    Contribution")
    print("-" * 65)
    
    weights = {
        "steps_completed": 0.5,
        "steps_in_progress": 0.10,
        "steps_available": 0.15,
        "steps_available_or_in_progress": 0.2,
        "operator_holding_similarity": 0.025,
        "gaze_target_similarity": 0.025
    }
    
    for metric, weight in weights.items():
        if metric in results:
            if metric.endswith("_similarity"):
                score = results[metric]
            else:
                score = results[metric].get("f1", 0.0)
            contribution = weight * score
            metric_label = metric.replace("_", " ").title()
            print(f"{metric_label:30} {weight:6.3f}    {score:5.3f}    {contribution:8.4f}")
        else:
            metric_label = metric.replace("_", " ").title()
            print(f"{metric_label:30} {weight:6.3f}    N/A      0.0000")
    
    print("\n" + "-"*50)
    print("STEPS ANALYSIS")
    print("-"*50)
    
    # Display all step metrics including the new union metric
    step_types = ["steps_completed", "steps_in_progress", "steps_available", "steps_available_or_in_progress"]
    step_labels = ["Steps Completed", "Steps In Progress", "Steps Available", "Steps Available OR In Progress"]
    
    for step_type, label in zip(step_types, step_labels):
        if step_type in results:
            step_metrics = results[step_type]
            print(f"\n{label}:")
            print(f"  F1 Score:  {step_metrics['f1']:.4f}")
            print(f"  Precision: {step_metrics['precision']:.4f}")
            print(f"  Recall:    {step_metrics['recall']:.4f}")
    
    print("\n" + "-"*50)
    print("TEXT SIMILARITY SCORES")
    print("-"*50)
    
    text_metrics = [
        ("current_keystep_similarity", "Current Keystep"),
        ("next_keystep_similarity", "Next Keystep"),
        ("operator_holding_similarity", "Operator Holding"),
        ("gaze_target_similarity", "Gaze Target")
    ]
    
    for metric, label in text_metrics:
        if metric in results:
            print(f"{label:20}: {results[metric]:.4f}")
    
    # Show Boolean States for reference (but note they're not in overall score)
    print("\n" + "-"*50)
    print("BOOLEAN STATES (Reference Only - Not in Overall Score)")
    print("-"*50)
    if "boolean_states" in results:
        bs = results["boolean_states"]
        print(f"F1 Score:     {bs['f1']:.4f}")
        print(f"Precision:    {bs['precision']:.4f}")
        print(f"Recall:       {bs['recall']:.4f}")
        print(f"Comparisons:  {bs['total_comparisons']}")
    else:
        print("No boolean states found in data.")
    
    print("\n" + "-"*50)
    print("TIMING ANALYSIS")
    print("-"*50)
    
    if "timing_statistics" in results:
        ts = results["timing_statistics"]
        print(f"Mean Frame Difference:   {ts['mean_difference']:.2f}")
        print(f"Std Frame Difference:    {ts['std_difference']:.2f}")
        print(f"Median Frame Difference: {ts['median_difference']:.2f}")
        print(f"Early Predictions:       {ts['early_predictions']}")
        print(f"Late Predictions:        {ts['late_predictions']}")
        print(f"Correct Predictions:     {ts['correct_predictions']}")
        print(f"Total Comparisons:       {ts['total_compared']}")
        
        if ts['total_compared'] > 0:
            accuracy = ts['correct_predictions'] / ts['total_compared']
            print(f"Timing Accuracy:         {accuracy:.4f}")
    
    if "timing_analysis" in results:
        print(f"\nDetailed Timing Analysis:")
        print(f"{'Step':<40} {'Pred':<8} {'Actual':<8} {'Diff':<8} {'Status'}")
        print("-" * 70)
        
        for step, analysis in results["timing_analysis"].items():
            pred = analysis.get("predicted_frame", "N/A")
            actual = analysis.get("actual_frame", "N/A")
            diff = analysis.get("difference")
            
            if diff is not None:
                if analysis["early"]:
                    status = "Early"
                elif analysis["late"]:
                    status = "Late"
                else:
                    status = "Correct"
                diff_str = f"{diff:+d}"
            else:
                status = "N/A"
                diff_str = "N/A"
            
            step_short = step[:39] if len(step) > 39 else step
            print(f"{step_short:<40} {str(pred):<8} {str(actual):<8} {diff_str:<8} {status}")

def main():
    """Main evaluation function."""
    # File paths
    icl_filename = "/home/mani/Repos/hcdt/logs/ICL_result_cooking_ex0001_no_exo_limited.json"
    gt_filename = "/home/mani/Repos/hcdt/data/Cooking/fair_cooking_05_02_gt.json"
    
    # Check if files exist
    if not os.path.exists(icl_filename):
        print(f"Error: ICL results file not found: {icl_filename}")
        return
    
    if not os.path.exists(gt_filename):
        print(f"Error: Ground truth file not found: {gt_filename}")
        return
    
    # Run evaluation
    try:
        results = evaluate_icl_results(icl_filename, gt_filename)
        print_evaluation_results(results)
        
        # Save results to file
        output_filename = "/home/mani/Repos/hcdt/eval/cooking_evaluation_results.json"
        with open(output_filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {output_filename}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
