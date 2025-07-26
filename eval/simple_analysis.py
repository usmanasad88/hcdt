"""
Simplified analysis script for ICL cooking evaluation results.
Focuses on detailed text analysis without visualizations.
"""

import json
import numpy as np
from collections import defaultdict, Counter
import os
from datetime import datetime

def load_evaluation_results(filename: str) -> dict:
    """Load evaluation results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def analyze_temporal_patterns(icl_filename: str, gt_filename: str):
    """Analyze temporal patterns in predictions vs ground truth."""
    
    # Load data
    with open(icl_filename, 'r') as f:
        icl_data = json.load(f)
    
    with open(gt_filename, 'r') as f:
        gt_data = json.load(f)
    
    # Create frame-indexed data
    icl_by_frame = {entry["frame_number"]: entry["state"] for entry in icl_data}
    gt_by_frame = {entry.get("frame_number", entry.get("frame", 0)): entry.get("state", {}) 
                   for entry in gt_data}
    
    # Analyze boolean state changes over time
    boolean_keys = [
        "pot_on_stove", "pot_burner_on", "water_in_pot", "water_boiling",
        "noodles_in_pot", "noodles_cooked", "noodles_drained", "cabbage_prepared",
        "garlic_prepared", "spring_onions_prepared", "celery_prepared",
        "pan_on_stove", "pan_burner_on", "oil_in_pan", "aromatics_in_pan",
        "noodles_in_pan", "sauce_in_pan", "stir_fry_cooked", "food_on_plate"
    ]
    
    # Track state transitions
    transitions = defaultdict(list)
    accuracy_over_time = []
    
    frames = sorted(set(icl_by_frame.keys()) & set(gt_by_frame.keys()))
    
    for frame in frames:
        icl_state = icl_by_frame[frame]
        gt_state = gt_by_frame[frame]
        
        # Calculate frame accuracy
        correct = 0
        total = 0
        
        for key in boolean_keys:
            if key in icl_state and key in gt_state:
                total += 1
                if icl_state[key] == gt_state[key]:
                    correct += 1
                
                # Track transitions
                transitions[key].append({
                    'frame': frame,
                    'predicted': icl_state[key],
                    'actual': gt_state[key],
                    'correct': icl_state[key] == gt_state[key]
                })
        
        if total > 0:
            accuracy_over_time.append({
                'frame': frame,
                'accuracy': correct / total,
                'correct': correct,
                'total': total
            })
    
    return {
        'transitions': transitions,
        'accuracy_over_time': accuracy_over_time,
        'frames_analyzed': len(frames)
    }

def analyze_step_progression(icl_filename: str, gt_filename: str):
    """Analyze how step progression differs between prediction and ground truth."""
    
    with open(icl_filename, 'r') as f:
        icl_data = json.load(f)
    
    with open(gt_filename, 'r') as f:
        gt_data = json.load(f)
    
    # Track step counts over time
    icl_step_counts = []
    gt_step_counts = []
    
    for entry in icl_data:
        frame = entry["frame_number"]
        state = entry["state"]
        
        icl_step_counts.append({
            'frame': frame,
            'completed': len(state.get('steps_completed', [])),
            'in_progress': len(state.get('steps_in_progress', [])),
            'available': len(state.get('steps_available', []))
        })
    
    # Create frame mapping for ground truth
    gt_by_frame = {entry.get("frame_number", entry.get("frame", 0)): entry.get("state", {}) 
                   for entry in gt_data}
    
    for entry in icl_data:
        frame = entry["frame_number"]
        if frame in gt_by_frame:
            gt_state = gt_by_frame[frame]
            gt_step_counts.append({
                'frame': frame,
                'completed': len(gt_state.get('steps_completed', [])),
                'in_progress': len(gt_state.get('steps_in_progress', [])),
                'available': len(gt_state.get('steps_available', []))
            })
    
    return {
        'icl_step_counts': icl_step_counts,
        'gt_step_counts': gt_step_counts
    }

def analyze_error_patterns(icl_filename: str, gt_filename: str):
    """Analyze common error patterns in predictions."""
    
    with open(icl_filename, 'r') as f:
        icl_data = json.load(f)
    
    with open(gt_filename, 'r') as f:
        gt_data = json.load(f)
    
    # Create frame-indexed data
    icl_by_frame = {entry["frame_number"]: entry["state"] for entry in icl_data}
    gt_by_frame = {entry.get("frame_number", entry.get("frame", 0)): entry.get("state", {}) 
                   for entry in gt_data}
    
    boolean_keys = [
        "pot_on_stove", "pot_burner_on", "water_in_pot", "water_boiling",
        "noodles_in_pot", "noodles_cooked", "noodles_drained", "cabbage_prepared",
        "garlic_prepared", "spring_onions_prepared", "celery_prepared",
        "pan_on_stove", "pan_burner_on", "oil_in_pan", "aromatics_in_pan",
        "noodles_in_pan", "sauce_in_pan", "stir_fry_cooked", "food_on_plate"
    ]
    
    error_patterns = defaultdict(lambda: {"false_positives": 0, "false_negatives": 0, "total": 0})
    step_errors = defaultdict(lambda: {"missed": 0, "extra": 0, "total": 0})
    
    frames = sorted(set(icl_by_frame.keys()) & set(gt_by_frame.keys()))
    
    for frame in frames:
        icl_state = icl_by_frame[frame]
        gt_state = gt_by_frame[frame]
        
        # Analyze boolean errors
        for key in boolean_keys:
            if key in icl_state and key in gt_state:
                error_patterns[key]["total"] += 1
                
                if icl_state[key] and not gt_state[key]:
                    error_patterns[key]["false_positives"] += 1
                elif not icl_state[key] and gt_state[key]:
                    error_patterns[key]["false_negatives"] += 1
        
        # Analyze step errors
        for step_type in ['steps_completed', 'steps_in_progress', 'steps_available']:
            icl_steps = set(step.lower().strip() for step in icl_state.get(step_type, []))
            gt_steps = set(step.lower().strip() for step in gt_state.get(step_type, []))
            
            missed = gt_steps - icl_steps
            extra = icl_steps - gt_steps
            
            step_errors[step_type]["missed"] += len(missed)
            step_errors[step_type]["extra"] += len(extra)
            step_errors[step_type]["total"] += 1
    
    return {
        'boolean_errors': error_patterns,
        'step_errors': step_errors
    }

def generate_comprehensive_report(results: dict, temporal_analysis: dict, step_analysis: dict, 
                                error_analysis: dict, output_filename: str):
    """Generate a comprehensive markdown report."""
    
    with open(output_filename, 'w') as f:
        f.write("# Comprehensive ICL Cooking Evaluation Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write(f"**Overall Score: {results.get('overall_score', 0.0):.4f}**\n\n")
        
        # Performance grade
        overall_score = results.get('overall_score', 0.0)
        if overall_score >= 0.9:
            grade = "A (Excellent)"
        elif overall_score >= 0.8:
            grade = "B (Good)"
        elif overall_score >= 0.7:
            grade = "C (Satisfactory)"
        elif overall_score >= 0.6:
            grade = "D (Needs Improvement)"
        else:
            grade = "F (Poor)"
        
        f.write(f"**Performance Grade: {grade}**\n\n")
        
        f.write("### Key Findings:\n")
        f.write(f"- Boolean state prediction F1: {results.get('boolean_states', {}).get('f1', 0.0):.4f}\n")
        f.write(f"- Steps completion F1: {results['steps_completed']['f1']:.4f}\n")
        f.write(f"- Current keystep similarity (BERT): {results['current_keystep_similarity']:.4f}\n")
        f.write(f"- Next keystep similarity (BERT): {results['next_keystep_similarity']:.4f}\n")
        
        if 'timing_statistics' in results:
            ts = results['timing_statistics']
            f.write(f"- Timing accuracy: {ts['correct_predictions']}/{ts['total_compared']} ({ts['correct_predictions']/ts['total_compared']*100:.1f}%)\n")
            f.write(f"- Average timing error: {ts['mean_difference']:.1f} frames\n")
        
        f.write("\n")
        
        # Strengths and Weaknesses
        f.write("### Strengths:\n")
        strengths = []
        # if results['boolean_states']['f1'] > 0.8:
        #     strengths.append("High accuracy in boolean state prediction")
        if results['current_keystep_similarity'] > 0.8:
            strengths.append("Good understanding of current keysteps")
        if results['steps_completed']['precision'] > 0.8:
            strengths.append("Low false positive rate for completed steps")
        
        if strengths:
            for strength in strengths:
                f.write(f"- {strength}\n")
        else:
            f.write("- Analysis reveals areas for improvement across all metrics\n")
        
        f.write("\n### Weaknesses:\n")
        weaknesses = []
        if results['steps_in_progress']['f1'] < 0.5:
            weaknesses.append("Poor identification of steps in progress")
        # if results['boolean_states']['recall'] < 0.7:
        #     weaknesses.append("Missing many true positive boolean states")
        if 'timing_statistics' in results and results['timing_statistics']['correct_predictions'] == 0:
            weaknesses.append("No correct timing predictions")
        
        if weaknesses:
            for weakness in weaknesses:
                f.write(f"- {weakness}\n")
        else:
            f.write("- Performance is generally strong across metrics\n")
        f.write("\n")
        
        # Detailed Metrics
        f.write("## Detailed Performance Metrics\n\n")
        
        # f.write("### Boolean States Performance\n")
        # bs = results['boolean_states']
        # f.write(f"| Metric | Score |\n")
        # f.write(f"|--------|-------|\n")
        # f.write(f"| F1 Score | {bs['f1']:.4f} |\n")
        # f.write(f"| Precision | {bs['precision']:.4f} |\n")
        # f.write(f"| Recall | {bs['recall']:.4f} |\n")
        # f.write(f"| Total Comparisons | {bs['total_comparisons']} |\n\n")
        
        f.write("### Steps Analysis\n")
        f.write("| Step Type | F1 | Precision | Recall |\n")
        f.write("|-----------|----|-----------|---------|\n")
        for step_type in ['steps_completed', 'steps_in_progress', 'steps_available']:
            metrics = results[step_type]
            label = step_type.replace('_', ' ').title()
            f.write(f"| {label} | {metrics['f1']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} |\n")
        f.write("\n")
        
        f.write("### Text Similarity Analysis (BERT Scores)\n")
        f.write("| Field | BERT Score | Interpretation |\n")
        f.write("|-------|------------|----------------|\n")
        
        text_metrics = [
            ('current_keystep_similarity', 'Current Keystep'),
            ('next_keystep_similarity', 'Next Keystep'),
            ('operator_holding_similarity', 'Operator Holding'),
            ('gaze_target_similarity', 'Gaze Target')
        ]
        
        for metric, label in text_metrics:
            score = results[metric]
            if score > 0.9:
                interp = "Excellent"
            elif score > 0.8:
                interp = "Good"
            elif score > 0.7:
                interp = "Fair"
            elif score > 0.6:
                interp = "Poor"
            else:
                interp = "Very Poor"
            f.write(f"| {label} | {score:.4f} | {interp} |\n")
        f.write("\n")
        
        # Error Analysis
        if error_analysis:
            f.write("## Error Pattern Analysis\n\n")
            
            f.write("### Boolean State Errors\n")
            f.write("| State | False Positives | False Negatives | Error Rate |\n")
            f.write("|-------|-----------------|-----------------|------------|\n")
            
            for state, errors in error_analysis['boolean_errors'].items():
                if errors['total'] > 0:
                    error_rate = (errors['false_positives'] + errors['false_negatives']) / errors['total']
                    f.write(f"| {state} | {errors['false_positives']} | {errors['false_negatives']} | {error_rate:.3f} |\n")
            f.write("\n")
            
            f.write("### Step Prediction Errors\n")
            f.write("| Step Type | Missed Steps | Extra Steps | Avg Error Rate |\n")
            f.write("|-----------|--------------|-------------|----------------|\n")
            
            for step_type, errors in error_analysis['step_errors'].items():
                if errors['total'] > 0:
                    avg_missed = errors['missed'] / errors['total']
                    avg_extra = errors['extra'] / errors['total']
                    label = step_type.replace('_', ' ').title()
                    f.write(f"| {label} | {avg_missed:.2f} | {avg_extra:.2f} | {(avg_missed + avg_extra):.2f} |\n")
            f.write("\n")
        
        # Timing Analysis
        if 'timing_statistics' in results:
            f.write("## Temporal Analysis\n\n")
            ts = results['timing_statistics']
            
            f.write("### Timing Statistics\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Mean frame difference | {ts['mean_difference']:.2f} |\n")
            f.write(f"| Standard deviation | {ts['std_difference']:.2f} |\n")
            f.write(f"| Median difference | {ts['median_difference']:.2f} |\n")
            f.write(f"| Early predictions | {ts['early_predictions']} |\n")
            f.write(f"| Late predictions | {ts['late_predictions']} |\n")
            f.write(f"| Correct timing | {ts['correct_predictions']} |\n")
            f.write(f"| Total comparisons | {ts['total_compared']} |\n\n")
            
            if 'timing_analysis' in results:
                f.write("### Detailed Step Timing Analysis\n")
                f.write("| Step | Predicted Frame | Actual Frame | Difference | Status |\n")
                f.write("|------|-----------------|--------------|------------|--------|\n")
                
                for step, analysis in results['timing_analysis'].items():
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
                    
                    step_short = step[:50] + "..." if len(step) > 50 else step
                    f.write(f"| {step_short} | {pred} | {actual} | {diff_str} | {status} |\n")
                f.write("\n")
        
        # Temporal Patterns
        if temporal_analysis['accuracy_over_time']:
            f.write("### Accuracy Trends\n")
            accuracies = [item['accuracy'] for item in temporal_analysis['accuracy_over_time']]
            frames = [item['frame'] for item in temporal_analysis['accuracy_over_time']]
            
            f.write(f"- Initial accuracy (first 10%): {np.mean(accuracies[:len(accuracies)//10]):.4f}\n")
            f.write(f"- Final accuracy (last 10%): {np.mean(accuracies[-len(accuracies)//10:]):.4f}\n")
            f.write(f"- Overall accuracy range: {min(accuracies):.4f} - {max(accuracies):.4f}\n")
            f.write(f"- Accuracy standard deviation: {np.std(accuracies):.4f}\n\n")
        
        # Recommendations
        f.write("## Actionable Recommendations\n\n")
        
        f.write("### High Priority:\n")
        recommendations = []
        
        if results['steps_in_progress']['f1'] < 0.5:
            recommendations.append("**Improve step transition detection**: The model struggles with identifying ongoing tasks. Consider adding temporal context windows or state transition models.")
        
        if 'timing_statistics' in results and results['timing_statistics']['early_predictions'] > results['timing_statistics']['late_predictions'] * 2:
            recommendations.append("**Address early prediction bias**: The model predicts step completion too early. Implement delay mechanisms or more conservative completion criteria.")
        
        # if results['boolean_states']['recall'] < 0.8:
        #     recommendations.append("**Reduce false negatives**: The model misses many true states. Consider lowering confidence thresholds or improving feature extraction.")
        
        if not recommendations:
            recommendations.append("**Fine-tune existing approach**: Performance is generally good. Focus on incremental improvements through hyperparameter tuning.")
        
        for rec in recommendations:
            f.write(f"- {rec}\n")
        
        f.write("\n### Medium Priority:\n")
        
        medium_recs = []
        if results['current_keystep_similarity'] < 0.8:
            medium_recs.append("**Improve keystep understanding**: Enhance natural language processing for better keystep prediction accuracy.")
        
        if results['steps_available']['f1'] < 0.7:
            medium_recs.append("**Better availability prediction**: Improve logic for determining which steps are currently available.")
        
        if not medium_recs:
            medium_recs.append("**Consider ensemble methods**: Combine multiple models or approaches for better overall performance.")
        
        for rec in medium_recs:
            f.write(f"- {rec}\n")
        
        f.write("\n### Low Priority:\n")
        f.write("- **Optimize computational efficiency**: Once accuracy targets are met, focus on speed improvements.\n")
        f.write("- **Expand evaluation metrics**: Consider additional evaluation criteria for comprehensive assessment.\n")
        f.write("- **Cross-validation**: Test performance on additional datasets for generalization assessment.\n\n")
        
        # Technical Implementation Suggestions
        f.write("## Technical Implementation Suggestions\n\n")
        
        if results['steps_in_progress']['f1'] < 0.5:
            f.write("### For Steps in Progress Detection:\n")
            f.write("1. **Temporal Modeling**: Implement LSTM/GRU layers to capture temporal dependencies\n")
            f.write("2. **State Transition Matrix**: Define explicit transition probabilities between states\n")
            f.write("3. **Multi-frame Context**: Use sliding windows of 3-5 frames for context\n")
            f.write("4. **Uncertainty Quantification**: Add confidence scores for in-progress predictions\n\n")
        
        # if results['boolean_states']['recall'] < 0.8:
        #     f.write("### For Boolean State Prediction:\n")
        #     f.write("1. **Feature Engineering**: Add more visual features (object detection, pose estimation)\n")
        #     f.write("2. **Data Augmentation**: Increase training data diversity\n")
        #     f.write("3. **Class Balancing**: Address imbalanced boolean state distributions\n")
        #     f.write("4. **Threshold Optimization**: Use ROC analysis to find optimal decision thresholds\n\n")
        
        if 'timing_statistics' in results and results['timing_statistics']['mean_difference'] < -1000:
            f.write("### For Timing Accuracy:\n")
            f.write("1. **Temporal Calibration**: Add learned delay parameters for each step type\n")
            f.write("2. **Progressive Completion**: Model steps as gradually completing rather than binary\n")
            f.write("3. **Context-Aware Timing**: Use video context to better estimate completion timing\n")
            f.write("4. **Post-processing Smoothing**: Apply temporal smoothing to reduce timing jitter\n\n")

def main():
    """Main function to run comprehensive analysis."""
    
    # File paths
    results_file = "/home/mani/Repos/hcdt/eval/cooking_evaluation_results.json"
    icl_file = "/home/mani/Repos/hcdt/logs/ICL_result_cooking_ex0001_no_exo_limited.json"
    gt_file = "/home/mani/Repos/hcdt/data/Cooking/fair_cooking_05_02_gt.json"
    output_dir = "/home/mani/Repos/hcdt/eval/analysis_output"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading evaluation results...")
    results = load_evaluation_results(results_file)
    
    print("Analyzing temporal patterns...")
    temporal_analysis = analyze_temporal_patterns(icl_file, gt_file)
    
    print("Analyzing step progression...")
    step_analysis = analyze_step_progression(icl_file, gt_file)
    
    print("Analyzing error patterns...")
    error_analysis = analyze_error_patterns(icl_file, gt_file)
    
    print("Generating comprehensive report...")
    generate_comprehensive_report(results, temporal_analysis, step_analysis, 
                                error_analysis, f"{output_dir}/comprehensive_report.md")
    
    # Save analysis data
    analysis_data = {
        'temporal_analysis': temporal_analysis,
        'step_analysis': step_analysis,
        'error_analysis': error_analysis
    }
    
    with open(f"{output_dir}/analysis_data.json", 'w') as f:
        json.dump(analysis_data, f, indent=2, default=str)
    
    print(f"\nComprehensive analysis complete!")
    print(f"Results saved in: {output_dir}")
    print(f"- Comprehensive report: {output_dir}/comprehensive_report.md")
    print(f"- Analysis data: {output_dir}/analysis_data.json")
    
    # Print summary
    print(f"\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Overall Score: {results.get('overall_score', 0.0):.4f}")
    print(f"Boolean States F1: {results['boolean_states']['f1']:.4f}")
    print(f"Steps Completed F1: {results['steps_completed']['f1']:.4f}")
    print(f"Steps In Progress F1: {results['steps_in_progress']['f1']:.4f}")
    print(f"Current Keystep Similarity: {results['current_keystep_similarity']:.4f}")
    
    if 'timing_statistics' in results:
        ts = results['timing_statistics']
        print(f"Timing Accuracy: {ts['correct_predictions']}/{ts['total_compared']} ({ts['correct_predictions']/ts['total_compared']*100:.1f}%)")

if __name__ == "__main__":
    main()
