"""
Advanced analysis script for ICL cooking evaluation results.
Provides detailed insights, visualizations, and recommendations.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict, Counter
import os
from datetime import datetime

# Try to import seaborn, if not available use matplotlib defaults
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

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
    icl_by_frame = {entry["frame"]: entry["state"] for entry in icl_data}
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
        frame = entry["frame"]
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
        frame = entry["frame"]
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

def create_visualizations(results: dict, temporal_analysis: dict, step_analysis: dict, output_dir: str):
    """Create comprehensive visualizations of the evaluation results."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    if SEABORN_AVAILABLE:
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    else:
        plt.style.use('default')
    
    # 1. Overall Metrics Bar Chart
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Boolean states metrics
    bs_metrics = results['boolean_states']
    ax1.bar(['F1', 'Precision', 'Recall'], 
            [bs_metrics['f1'], bs_metrics['precision'], bs_metrics['recall']])
    ax1.set_title('Boolean States Performance')
    ax1.set_ylabel('Score')
    ax1.set_ylim(0, 1)
    
    # Steps metrics
    step_types = ['steps_completed', 'steps_in_progress', 'steps_available']
    step_f1s = [results[st]['f1'] for st in step_types]
    
    ax2.bar([st.replace('_', ' ').title() for st in step_types], step_f1s)
    ax2.set_title('Steps F1 Scores')
    ax2.set_ylabel('F1 Score')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45)
    
    # Text similarity metrics
    text_metrics = ['current_keystep_similarity', 'next_keystep_similarity', 
                   'operator_holding_similarity', 'gaze_target_similarity']
    text_scores = [results[tm] for tm in text_metrics]
    text_labels = [tm.replace('_similarity', '').replace('_', ' ').title() for tm in text_metrics]
    
    ax3.bar(text_labels, text_scores)
    ax3.set_title('Text Similarity Scores (BERT)')
    ax3.set_ylabel('BERT Score')
    ax3.set_ylim(0, 1)
    ax3.tick_params(axis='x', rotation=45)
    
    # Timing analysis
    if 'timing_statistics' in results:
        ts = results['timing_statistics']
        timing_labels = ['Early', 'Late', 'Correct']
        timing_counts = [ts['early_predictions'], ts['late_predictions'], ts['correct_predictions']]
        
        ax4.pie(timing_counts, labels=timing_labels, autopct='%1.1f%%')
        ax4.set_title('Timing Accuracy Distribution')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/overall_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Accuracy over time
    if temporal_analysis['accuracy_over_time']:
        fig, ax = plt.subplots(figsize=(15, 6))
        
        frames = [item['frame'] for item in temporal_analysis['accuracy_over_time']]
        accuracies = [item['accuracy'] for item in temporal_analysis['accuracy_over_time']]
        
        ax.plot(frames, accuracies, linewidth=2, marker='o', markersize=2)
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Boolean State Accuracy')
        ax.set_title('Prediction Accuracy Over Time')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(frames, accuracies, 1)
        p = np.poly1d(z)
        ax.plot(frames, p(frames), "--", alpha=0.8, color='red', 
                label=f'Trend (slope: {z[0]:.6f})')
        ax.legend()
        
        plt.savefig(f'{output_dir}/accuracy_over_time.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Step progression comparison
    if step_analysis['icl_step_counts'] and step_analysis['gt_step_counts']:
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        icl_df = pd.DataFrame(step_analysis['icl_step_counts'])
        gt_df = pd.DataFrame(step_analysis['gt_step_counts'])
        
        step_types = ['completed', 'in_progress', 'available']
        titles = ['Steps Completed', 'Steps In Progress', 'Steps Available']
        
        for i, (step_type, title) in enumerate(zip(step_types, titles)):
            axes[i].plot(icl_df['frame'], icl_df[step_type], 
                        label='Predicted', linewidth=2, marker='o', markersize=1)
            axes[i].plot(gt_df['frame'], gt_df[step_type], 
                        label='Ground Truth', linewidth=2, marker='s', markersize=1)
            axes[i].set_title(title)
            axes[i].set_ylabel('Count')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Frame Number')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/step_progression.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Timing analysis histogram
    if 'timing_analysis' in results:
        timing_diffs = []
        for step, analysis in results['timing_analysis'].items():
            if analysis['difference'] is not None:
                timing_diffs.append(analysis['difference'])
        
        if timing_diffs:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            ax.hist(timing_diffs, bins=20, alpha=0.7, edgecolor='black')
            ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Perfect Timing')
            ax.axvline(x=np.mean(timing_diffs), color='orange', linestyle='-', 
                      linewidth=2, label=f'Mean: {np.mean(timing_diffs):.1f}')
            
            ax.set_xlabel('Frame Difference (Predicted - Actual)')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Timing Errors')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.savefig(f'{output_dir}/timing_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()

def generate_detailed_report(results: dict, temporal_analysis: dict, step_analysis: dict, 
                           output_filename: str):
    """Generate a comprehensive markdown report."""
    
    with open(output_filename, 'w') as f:
        f.write("# ICL Cooking Evaluation Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write(f"**Overall Score: {results.get('overall_score', 0.0):.4f}**\n\n")
        
        f.write("### Key Findings:\n")
        f.write(f"- Boolean state prediction F1: {results['boolean_states']['f1']:.4f}\n")
        f.write(f"- Steps completion F1: {results['steps_completed']['f1']:.4f}\n")
        f.write(f"- Text similarity (BERT): {results['current_keystep_similarity']:.4f}\n")
        
        if 'timing_statistics' in results:
            ts = results['timing_statistics']
            f.write(f"- Timing accuracy: {ts['correct_predictions']}/{ts['total_compared']} ({ts['correct_predictions']/ts['total_compared']*100:.1f}%)\n")
        
        f.write("\n")
        
        # Detailed Metrics
        f.write("## Detailed Performance Metrics\n\n")
        
        f.write("### Boolean States\n")
        bs = results['boolean_states']
        f.write(f"- F1 Score: {bs['f1']:.4f}\n")
        f.write(f"- Precision: {bs['precision']:.4f}\n")
        f.write(f"- Recall: {bs['recall']:.4f}\n")
        f.write(f"- Total Comparisons: {bs['total_comparisons']}\n\n")
        
        f.write("### Steps Analysis\n")
        for step_type in ['steps_completed', 'steps_in_progress', 'steps_available']:
            metrics = results[step_type]
            f.write(f"**{step_type.replace('_', ' ').title()}:**\n")
            f.write(f"- F1: {metrics['f1']:.4f}\n")
            f.write(f"- Precision: {metrics['precision']:.4f}\n")
            f.write(f"- Recall: {metrics['recall']:.4f}\n\n")
        
        f.write("### Text Similarity (BERT Scores)\n")
        text_metrics = ['current_keystep_similarity', 'next_keystep_similarity', 
                       'operator_holding_similarity', 'gaze_target_similarity']
        for metric in text_metrics:
            label = metric.replace('_similarity', '').replace('_', ' ').title()
            f.write(f"- {label}: {results[metric]:.4f}\n")
        f.write("\n")
        
        # Timing Analysis
        if 'timing_statistics' in results:
            f.write("### Timing Analysis\n")
            ts = results['timing_statistics']
            f.write(f"- Mean frame difference: {ts['mean_difference']:.2f}\n")
            f.write(f"- Standard deviation: {ts['std_difference']:.2f}\n")
            f.write(f"- Early predictions: {ts['early_predictions']}\n")
            f.write(f"- Late predictions: {ts['late_predictions']}\n")
            f.write(f"- Correct timing: {ts['correct_predictions']}\n\n")
        
        # Recommendations
        f.write("## Recommendations for Improvement\n\n")
        
        # Based on results, provide specific recommendations
        if results['steps_in_progress']['f1'] < 0.5:
            f.write("- **Steps in Progress**: Low F1 score suggests difficulty in identifying ongoing tasks. Consider improving temporal reasoning.\n")
        
        if results['boolean_states']['recall'] < results['boolean_states']['precision']:
            f.write("- **Boolean States**: Low recall indicates missing true positives. The model may be too conservative.\n")
        
        if 'timing_statistics' in results and results['timing_statistics']['early_predictions'] > results['timing_statistics']['late_predictions']:
            f.write("- **Timing**: Model tends to predict step completion too early. Consider adding delay mechanisms.\n")
        
        f.write("- Consider ensemble methods to improve overall performance.\n")
        f.write("- Focus on temporal consistency across predictions.\n")

def main():
    """Main function to run advanced analysis."""
    
    # File paths
    results_file = "/home/mani/Repos/hcdt/eval/cooking_evaluation_results.json"
    icl_file = "/home/mani/Repos/hcdt/logs/ICL_result_cooking_ex0001_no_exo_limited.json"
    gt_file = "/home/mani/Repos/hcdt/data/Cooking/fair_cooking_05_02_gt.json"
    output_dir = "/home/mani/Repos/hcdt/eval/analysis_output"
    
    print("Loading evaluation results...")
    results = load_evaluation_results(results_file)
    
    print("Analyzing temporal patterns...")
    temporal_analysis = analyze_temporal_patterns(icl_file, gt_file)
    
    print("Analyzing step progression...")
    step_analysis = analyze_step_progression(icl_file, gt_file)
    
    print("Creating visualizations...")
    create_visualizations(results, temporal_analysis, step_analysis, output_dir)
    
    print("Generating detailed report...")
    generate_detailed_report(results, temporal_analysis, step_analysis, 
                           f"{output_dir}/detailed_report.md")
    
    print(f"\nAdvanced analysis complete!")
    print(f"Results saved in: {output_dir}")
    print(f"- Visualizations: {output_dir}/*.png")
    print(f"- Detailed report: {output_dir}/detailed_report.md")

if __name__ == "__main__":
    main()
