"""
Main evaluation script for ICL cooking results.
This script provides a complete evaluation pipeline with both basic and advanced analysis.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our evaluation modules
from evaluate_icl_cooking import evaluate_icl_results, print_evaluation_results
from simple_analysis import (analyze_temporal_patterns, analyze_step_progression, 
                           analyze_error_patterns, generate_comprehensive_report)

def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(description='Evaluate ICL cooking results against ground truth')
    parser.add_argument('--icl_file', type=str, 
                       default='logs/ICL_result_cooking_ex0001_no_ego_no_gaze.json',
                       help='Path to ICL results JSON file')
    parser.add_argument('--gt_file', type=str,
                       default='/home/mani/Repos/hcdt/data/Cooking/fair_cooking_05_02_gt.json',
                       help='Path to ground truth JSON file')
    parser.add_argument('--output_dir', type=str,
                       default='/home/mani/Repos/hcdt/eval/evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--skip_bert', action='store_true',
                       help='Skip BERT score computation for faster evaluation')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("ICL COOKING EVALUATION PIPELINE")
    print("="*80)
    print(f"ICL Results: {args.icl_file}")
    print(f"Ground Truth: {args.gt_file}")
    print(f"Output Directory: {args.output_dir}")
    print()
    
    # Check if files exist
    if not os.path.exists(args.icl_file):
        print(f"Error: ICL results file not found: {args.icl_file}")
        return 1
    
    if not os.path.exists(args.gt_file):
        print(f"Error: Ground truth file not found: {args.gt_file}")
        return 1
    
    try:
        # Step 1: Basic evaluation
        print("Step 1: Running basic evaluation...")
        results = evaluate_icl_results(args.icl_file, args.gt_file)
        
        # Save basic results
        import json
        basic_results_file = os.path.join(args.output_dir, "basic_evaluation_results.json")
        with open(basic_results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("Step 2: Printing basic results...")
        print_evaluation_results(results)
        
        # Step 3: Advanced analysis
        print("\nStep 3: Running advanced analysis...")
        temporal_analysis = analyze_temporal_patterns(args.icl_file, args.gt_file)
        step_analysis = analyze_step_progression(args.icl_file, args.gt_file)
        error_analysis = analyze_error_patterns(args.icl_file, args.gt_file)
        
        # Step 4: Generate comprehensive report
        print("Step 4: Generating comprehensive report...")
        report_file = os.path.join(args.output_dir, "comprehensive_evaluation_report.md")
        generate_comprehensive_report(results, temporal_analysis, step_analysis, 
                                    error_analysis, report_file)
        
        # Step 5: Save all analysis data
        print("Step 5: Saving analysis data...")
        analysis_data = {
            'basic_results': results,
            'temporal_analysis': temporal_analysis,
            'step_analysis': step_analysis,
            'error_analysis': error_analysis
        }
        
        analysis_file = os.path.join(args.output_dir, "complete_analysis_data.json")
        with open(analysis_file, 'w') as f:
            json.dump(analysis_data, f, indent=2, default=str)
        
        # Step 6: Create summary
        print("Step 6: Creating executive summary...")
        create_executive_summary(results, args.output_dir)
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)
        print(f"📊 Overall Score: {results.get('overall_score', 0.0):.4f}")
        print(f"📁 Results saved in: {args.output_dir}")
        print("📋 Generated files:")
        print(f"   - {basic_results_file}")
        print(f"   - {report_file}")
        print(f"   - {analysis_file}")
        print(f"   - {os.path.join(args.output_dir, 'executive_summary.txt')}")
        
        return 0
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1

def create_executive_summary(results: dict, output_dir: str):
    """Create a brief executive summary."""
    
    summary_file = os.path.join(output_dir, "executive_summary.txt")
    
    with open(summary_file, 'w') as f:
        f.write("ICL COOKING EVALUATION - EXECUTIVE SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        # Overall performance
        overall_score = results.get('overall_score', 0.0)
        f.write(f"OVERALL PERFORMANCE SCORE: {overall_score:.4f}\n\n")
        
        # Performance grade
        if overall_score >= 0.9:
            grade = "A (Excellent)"
            recommendation = "Deploy with confidence"
        elif overall_score >= 0.8:
            grade = "B (Good)" 
            recommendation = "Minor improvements needed"
        elif overall_score >= 0.7:
            grade = "C (Satisfactory)"
            recommendation = "Moderate improvements needed"
        elif overall_score >= 0.6:
            grade = "D (Needs Improvement)"
            recommendation = "Significant improvements required"
        else:
            grade = "F (Poor)"
            recommendation = "Major redesign required"
        
        f.write(f"PERFORMANCE GRADE: {grade}\n")
        f.write(f"RECOMMENDATION: {recommendation}\n\n")
        
        # Key metrics
        f.write("KEY METRICS:\n")
        f.write(f"• Boolean States F1: {results['boolean_states']['f1']:.3f}\n")
        f.write(f"• Steps Completed F1: {results['steps_completed']['f1']:.3f}\n")
        f.write(f"• Steps In Progress F1: {results['steps_in_progress']['f1']:.3f}\n")
        f.write(f"• Current Keystep Similarity: {results['current_keystep_similarity']:.3f}\n")
        
        if 'timing_statistics' in results:
            ts = results['timing_statistics']
            accuracy = ts['correct_predictions'] / ts['total_compared'] if ts['total_compared'] > 0 else 0
            f.write(f"• Timing Accuracy: {accuracy:.1%}\n")
        
        f.write("\nTOP ISSUES TO ADDRESS:\n")
        issues = []
        
        if results['steps_in_progress']['f1'] < 0.5:
            issues.append("• Poor detection of steps in progress")
        
        if 'timing_statistics' in results and results['timing_statistics']['correct_predictions'] == 0:
            issues.append("• No accurate timing predictions")
        
        if results['boolean_states']['recall'] < 0.8:
            issues.append("• Missing many true boolean states")
        
        if not issues:
            issues.append("• Performance is generally acceptable")
        
        for issue in issues[:3]:  # Top 3 issues
            f.write(f"{issue}\n")
        
        f.write(f"\nFor detailed analysis, see: comprehensive_evaluation_report.md\n")

if __name__ == "__main__":
    exit(main())
