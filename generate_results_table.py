# Read all results from the results directory and generate a LaTeX table summarizing the performance of different prompting strategies across various tasks.


# Sample table structure:
# \begin{table}[H]
# \centering
#     \caption{Task State Prediction Performance on Use Case Scenarios with Varying Contextual Inputs}
#     \label{tab:phase_one_results}
#     \begin{threeparttable}
#     \resizebox{\textwidth}{!}{%
#     \begin{tabular}{@{}l ccc cc@{}}
#     \toprule
#     \multirow{3}{*}{\textbf{Prompting Strategy}} & \multicolumn{3}{c}{\textbf{Mean F1 Score}} & \multicolumn{2}{c}{\textbf{Overall}} \\
#     \cmidrule(lr){2-4} \cmidrule(lr){5-6}
#     & Stacking & Assembly & Cooking & Score & Rank \\
#     \midrule
#     Current Image Only \\
#     \quad Gemini 2.5 Flash &  &  &  &  & 10 \\
#     \quad Gemma-27B &  &  &  &  & 9 \\
# Rolling Context Window (RCW-PS) + Prev. Predicted State \\
#     \quad Gemini 2.5 Flash &  &  &  &  & 6 \\
#     \quad Gemini 2.5 Flash Lite &  &  &  &  & 5 \\
#     \quad Gemma-27B &  &  &  &  & 3 \\

#     Rolling Context Window + Prev. Ground Truth State (RCW-GT) \\
#     \quad Gemini 2.5 Flash & 0.767 & 0.745 &  &  & 4 \\
#     \quad Gemini 2.5 Flash Lite & 0.674  & 0.738 &  &  & 3 \\
#     \quad Gemma-27B & 0.662 & 0.739 &  &  & 3 \\
#
#     In-Context Learning + History of Imgs \\
#     \quad Gemini Gemini 2.5 Flash Lite (no gaze) &  &  &  &  & \textbf{1} \\
#     \quad Gemini Gemini 2.5 Flash Lite (no ego) & - & - &  &  & \textbf{1} \\
#     \quad Gemini 2.5 Flash (Complete) &  &  &  &  & 2 \\
#     \bottomrule
#     \end{tabular}
#     }
#     \end{threeparttable}
# \end{table}

import os
import json
import glob
import re
from collections import defaultdict
from eval.run_evaluation import do_evaluation
from eval.temporal_f1_state_eval import load_json_data, calculate_segmental_f1, calculate_framewise_f1
import tempfile
from eval.visualize_task_states import visualize_comparison_timeline

# Ground truth file mappings
gt_files = {
    'Cooking': '/home/mani/Repos/hcdt/data/Cooking/fair_cooking_05_02_gt_final.json',
    'Stack': '/home/mani/Repos/hcdt/data/Stack/exp2_gt_final.json',
    'Stack_v2': '/home/mani/Repos/hcdt/data/Stack/exp2_gt_final.json',

    'HAViD': '/home/mani/Repos/hcdt/data/HAViD/S02A08I21_gt_final.json'
}

def parse_filename(filename):
    """
    Parse result filename to extract experiment details.
    Expected format: {EXP_TYPE}_{CASE_STUDY}_{MODEL}_{GAZE}_{GT}_result.json
    """
    basename = os.path.basename(filename)
    parts = basename.replace('_result.json', '').split('_')
    
    result = {
        'exp_type': None,
        'case_study': None,
        'model': None,
        'use_gaze': False,
        'use_gt': False,
        'use_ego': False
    }
    
    if len(parts) >= 3:
        result['exp_type'] = parts[0]  # ICL, RCWPS, etc.
        result['case_study'] = parts[1]  # Cooking, Stack_v2, havid
        
        # Handle compound model names like gemini-2.5-flash-lite-preview-06-17
        model_parts = []
        gaze_gt_parts = []
        
        # Find where model name ends and gaze/gt flags begin
        for i, part in enumerate(parts[2:], 2):
            if part in ['use', 'no'] or part.startswith('use_') or part.startswith('no_'):
                gaze_gt_parts = parts[i:]
                break
            model_parts.append(part)
        
        result['model'] = '_'.join(model_parts) if model_parts else parts[2]
        
        # Parse gaze and gt flags
        flags_str = '_'.join(gaze_gt_parts)
        result['use_gaze'] = 'use_gaze' in flags_str
        result['use_gt'] = 'use_gt' in flags_str
        result['use_ego'] = 'use_ego' in flags_str
    
    return result

def get_evaluation_score(result_file, gt_file):
    """
    Run evaluation on a result file and return the temporal F1 score.
    """
    try:
        # Load data using the existing function
        gt_data = load_json_data(gt_file)
        pred_data = load_json_data(result_file)
        
        if gt_data is None or pred_data is None:
            return 0.0

        # Configuration (matching the temporal_f1_state_eval.py)
        IOU_THRESHOLD = 0.5
        MERGE_GAP_FRAMES = 50  # Using the value from main() function
        
        # Calculate individual F1 scores using existing functions
        f1_completed, _, _ = calculate_segmental_f1(
            gt_data, pred_data, 'steps_completed', IOU_THRESHOLD, MERGE_GAP_FRAMES
        )
        
        f1_in_progress, _, _ = calculate_segmental_f1(
            gt_data, pred_data, 'steps_in_progress', IOU_THRESHOLD, MERGE_GAP_FRAMES
        )
        
        f1_available, _, _ = calculate_framewise_f1(gt_data, pred_data)
        
        # Calculate weighted score using the same weights as in temporal_f1_state_eval.py
        weights = {'completed': 0.5, 'in_progress': 0.3, 'available': 0.2}
        
        combined_f1 = (weights['completed'] * f1_completed +
                       weights['in_progress'] * f1_in_progress +
                       weights['available'] * f1_available)
        
        return combined_f1
    except Exception as e:
        print(f"Error evaluating {result_file}: {e}")
        return 0.0

def collect_results(logs_directory):
    """
    Collect all result files and their evaluation scores.
    """
    results = []
    unused_files = []
    
    # Find all result files (excluding phase2 results)
    result_files = glob.glob(os.path.join(logs_directory, "*_result.json"))
    
    for result_file in result_files:
        # Skip phase2 results
        if 'phase2' in os.path.basename(result_file).lower():
            unused_files.append({
                'filename': os.path.basename(result_file),
                'reason': 'Phase2 result (excluded by request)'
            })
            continue
            
        print(f"Processing {result_file}...")
        
        # Parse filename
        file_info = parse_filename(result_file)
        
        # Map case study names
        case_study = file_info['case_study']
        if case_study == 'HAViD':
            case_study_display = 'Assembly'
            gt_key = 'HAViD'
        elif case_study == 'Stack':
            case_study_display = 'Stacking'
            gt_key = 'Stack_v2'
        elif case_study == 'Cooking':
            case_study_display = 'Cooking'
            gt_key = 'Cooking'
        else:
            print(f"Unknown case study: {case_study}")
            unused_files.append({
                'filename': os.path.basename(result_file),
                'reason': f'Unknown case study: {case_study}'
            })
            continue
        
        # Get ground truth file
        gt_file = gt_files.get(gt_key)
        if not gt_file or not os.path.exists(gt_file):
            print(f"Ground truth file not found for {gt_key}")
            unused_files.append({
                'filename': os.path.basename(result_file),
                'reason': f'Ground truth file not found for {gt_key}'
            })
            continue
        
        # Get evaluation score
        score = get_evaluation_score(result_file, gt_file)
        
        results.append({
            'exp_type': file_info['exp_type'],
            'case_study': case_study_display,
            'model': file_info['model'],
            'use_gaze': file_info['use_gaze'],
            'use_gt': file_info['use_gt'],
            'use_ego': file_info['use_ego'],
            'score': score,
            'filename': os.path.basename(result_file)
        })
        
        print(f"  Score: {score:.3f}")
    
    return results, unused_files

def format_model_name(model):
    """Format model name for display in table."""
    if 'gemini' in model.lower():
        if '2.5' in model and 'flash' in model:
            if 'lite' in model:
                return 'Gemini 2.5 Flash Lite'
            else:
                return 'Gemini 2.5 Flash'
        return 'Gemini'
    elif 'gemma' in model.lower():
        if '27b' in model.lower():
            return 'Gemma-27B'
        return 'Gemma'
    return model

def organize_results_for_table(results):
    """
    Organize results into a structure suitable for LaTeX table generation.
    """
    # Group results by experiment type and configuration
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    
    for result in results:
        exp_type = result['exp_type']
        model = format_model_name(result['model'])
        case_study = result['case_study']
        use_gaze = result['use_gaze']
        use_gt = result['use_gt']
        use_ego = result['use_ego']
        
        # Create configuration key
        if exp_type == 'ICL':
            if not use_gaze and not use_ego:
                config = "In-Context Learning + History of Imgs (no gaze, no ego)"
            elif not use_gaze:
                config = "In-Context Learning + History of Imgs (no gaze)"
            elif not use_ego:
                config = "In-Context Learning + History of Imgs (no ego)"
            else:
                config = "In-Context Learning + History of Imgs (Complete)"
        elif exp_type == 'RCWPS':
            if use_gt:
                base_config = "Rolling Context Window + Prev. Ground Truth State (RCW-GT)"
            else:
                base_config = "Rolling Context Window (RCW-PS) + Prev. Predicted State"
            
            # Add ablation details for RCWPS/RCWGT
            ablations = []
            if not use_gaze:
                ablations.append("no gaze")
            if not use_ego:
                ablations.append("no ego")
            
            if ablations:
                config = f"{base_config} ({', '.join(ablations)})"
            else:
                config = base_config
        else:
            # Handle other experiment types (like direct image-only approaches)
            ablations = []
            if not use_gaze:
                ablations.append("no gaze")
            if not use_ego:
                ablations.append("no ego")
            
            if ablations:
                config = f"Current Image Only ({', '.join(ablations)})"
            else:
                config = "Current Image Only"
        
        grouped[config][model][case_study] = result['score']
    
    return grouped

def calculate_overall_scores_and_ranks(grouped):
    """
    Calculate overall scores and ranks for each model configuration.
    """
    model_scores = {}
    
    for config, models in grouped.items():
        for model, case_studies in models.items():
            key = (config, model)
            scores = list(case_studies.values())
            if scores:
                model_scores[key] = sum(scores) / len(scores)
            else:
                model_scores[key] = 0.0
    
    # Sort by score and assign ranks
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    ranks = {}
    for i, (key, score) in enumerate(sorted_models):
        ranks[key] = i + 1
    
    return model_scores, ranks

def generate_latex_table(grouped, model_scores, ranks):
    """
    Generate the LaTeX table string.
    """
    latex = []
    
    latex.append(r"\begin{table}[H]")
    latex.append(r"\centering")
    latex.append(r"    \caption{Task State Prediction Performance on Use Case Scenarios with Varying Contextual Inputs}")
    latex.append(r"    \label{tab:phase_one_results}")
    latex.append(r"    \begin{threeparttable}")
    latex.append(r"    \resizebox{\textwidth}{!}{%")
    latex.append(r"    \begin{tabular}{@{}l ccc cc@{}}")
    latex.append(r"    \toprule")
    latex.append(r"    \multirow{3}{*}{\textbf{Prompting Strategy}} & \multicolumn{3}{c}{\textbf{Mean F1 Score}} & \multicolumn{2}{c}{\textbf{Overall}} \\")
    latex.append(r"    \cmidrule(lr){2-4} \cmidrule(lr){5-6}")
    latex.append(r"    & Stacking & Assembly & Cooking & Score & Rank \\")
    latex.append(r"    \midrule")
    
    # Define the order of configurations (updated to include ablations)
    config_order = [
        "Current Image Only",
        "Current Image Only (no gaze)",
        "Current Image Only (no ego)",
        "Current Image Only (no gaze, no ego)",
        "Rolling Context Window (RCW-PS) + Prev. Predicted State",
        "Rolling Context Window (RCW-PS) + Prev. Predicted State (no gaze)",
        "Rolling Context Window (RCW-PS) + Prev. Predicted State (no ego)",
        "Rolling Context Window (RCW-PS) + Prev. Predicted State (no gaze, no ego)",
        "Rolling Context Window + Prev. Ground Truth State (RCW-GT)",
        "Rolling Context Window + Prev. Ground Truth State (RCW-GT) (no gaze)",
        "Rolling Context Window + Prev. Ground Truth State (RCW-GT) (no ego)",
        "Rolling Context Window + Prev. Ground Truth State (RCW-GT) (no gaze, no ego)",
        "In-Context Learning + History of Imgs (Complete)",
        "In-Context Learning + History of Imgs (no gaze)",
        "In-Context Learning + History of Imgs (no ego)",
        "In-Context Learning + History of Imgs (no gaze, no ego)"
    ]
    
    for config in config_order:
        if config not in grouped:
            continue
            
        latex.append(f"    {config} \\\\")
        
        models = grouped[config]
        # Sort models for consistent ordering
        for model in sorted(models.keys()):
            case_studies = models[model]
            
            stacking_score = case_studies.get('Stacking', 0)
            assembly_score = case_studies.get('Assembly', 0) 
            cooking_score = case_studies.get('Cooking', 0)
            
            overall_score = model_scores.get((config, model), 0)
            rank = ranks.get((config, model), '-')
            
            # Format scores (show only if > 0)
            stacking_str = f"{stacking_score:.3f}" if stacking_score > 0 else ""
            assembly_str = f"{assembly_score:.3f}" if assembly_score > 0 else ""
            cooking_str = f"{cooking_score:.3f}" if cooking_score > 0 else ""
            overall_str = f"{overall_score:.3f}" if overall_score > 0 else ""
            
            # Bold the rank if it's 1 or 2
            rank_str = f"\\textbf{{{rank}}}" if rank in [1, 2] else str(rank)
            
            latex.append(f"        \\quad {model} & {stacking_str} & {assembly_str} & {cooking_str} & {overall_str} & {rank_str} \\\\")
    
    latex.append(r"    \bottomrule")
    latex.append(r"    \end{tabular}")
    latex.append(r"    }")
    latex.append(r"    \end{threeparttable}")
    latex.append(r"\end{table}")
    
    return '\n'.join(latex)

def generate_visualizations(results, output_dir="/home/mani/Repos/hcdt/visualizations"):
    """
    Generate visualization plots for all result files.
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for result in results:
        case_study = result['case_study']
        filename = result['filename']
        
        # Map case study to ground truth file
        if case_study == 'Assembly':
            gt_key = 'HAViD'
        elif case_study == 'Stacking':
            gt_key = 'Stack_v2'
        elif case_study == 'Cooking':
            gt_key = 'Cooking'
        else:
            print(f"Skipping visualization for unknown case study: {case_study}")
            continue
        
        gt_file = gt_files.get(gt_key)
        if not gt_file or not os.path.exists(gt_file):
            print(f"Skipping visualization: Ground truth file not found for {gt_key}")
            continue
        
        # Construct full path to result file
        result_file = os.path.join("/home/mani/Repos/hcdt/logs/Publishable", filename)
        
        # Generate output filename (replace .json with .png)
        output_filename = filename.replace('_result.json', '_visualization.png')
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"Generating visualization for {filename}...")
        
        try:
            visualize_comparison_timeline(gt_file, result_file, output_path)
            print(f"  Saved to: {output_path}")
        except Exception as e:
            print(f"  Error generating visualization: {e}")

def main():
    """Main function to generate the results table."""
    logs_directory = "/home/mani/Repos/hcdt/logs/Publishable"
    
    print("Collecting results and running evaluations...")
    results, unused_files = collect_results(logs_directory)
    
    if not results:
        print("No valid result files found!")
        return
    
    print(f"\nFound {len(results)} result files")
    
    # Print unused files
    if unused_files:
        print(f"\nUnused files ({len(unused_files)}):")
        for unused in unused_files:
            print(f"  - {unused['filename']}: {unused['reason']}")
    
    print("Organizing results...")
    
    grouped = organize_results_for_table(results)
    model_scores, ranks = calculate_overall_scores_and_ranks(grouped)
    
    print("Generating LaTeX table...")
    latex_table = generate_latex_table(grouped, model_scores, ranks)
    
    # Save to file
    output_file = "/home/mani/Repos/hcdt/results_table.tex"
    with open(output_file, 'w') as f:
        f.write(latex_table)
    
    print(f"\nLaTeX table saved to: {output_file}")
    print("\nTable content:")
    print(latex_table)
    
    # Generate visualizations for all results
    print("\nGenerating visualizations...")
    generate_visualizations(results)
    
    # Also save a summary of results
    summary_file = "/home/mani/Repos/hcdt/results_summary.json"
    summary = {
        'results': results,
        'unused_files': unused_files,
        'model_scores': {f"{k[0]} - {k[1]}": v for k, v in model_scores.items()},
        'ranks': {f"{k[0]} - {k[1]}": v for k, v in ranks.items()}
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results summary saved to: {summary_file}")

if __name__ == "__main__":
    main()