#!/usr/bin/env python3
"""
Experiment Log Analyzer
Utility to analyze and compare experiment logs
"""

import json
import glob
import argparse
from pathlib import Path
from typing import List, Dict
import pandas as pd
from datetime import datetime


def load_experiment_logs(log_dir: str = ".") -> List[Dict]:
    """Load all experiment log files from directory"""
    log_files = glob.glob(f"{log_dir}/exp_*_log.json")
    experiments = []
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                exp_data = json.load(f)
                experiments.append(exp_data)
        except Exception as e:
            print(f"Warning: Could not load {log_file}: {e}")
    
    return experiments


def analyze_experiments(experiments: List[Dict]) -> None:
    """Analyze and compare experiments"""
    if not experiments:
        print("No experiment logs found.")
        return
    
    print("🔍 EXPERIMENT ANALYSIS")
    print("=" * 50)
    
    # Create comparison table
    data = []
    for exp in experiments:
        data.append({
            'Experiment ID': exp['experiment_id'],
            'Model': exp['config'].get('model', 'Unknown'),
            'Case Study': exp['config'].get('case_study', 'Unknown'),
            'Start Time': exp['start_time'][:19],  # Remove microseconds
            'Duration (min)': f"{exp.get('total_duration_seconds', 0) / 60:.1f}",
            'Generations': exp['total_generations'],
            'Total Tokens': f"{exp['total_tokens']:,}",
            'Avg Duration (s)': f"{sum(m['duration_seconds'] for m in exp['metrics']) / len(exp['metrics']) if exp['metrics'] else 0:.2f}",
            'Est. Cost ($)': f"{exp['total_input_tokens'] * 0.00001875 + exp['total_output_tokens'] * 0.000075:.4f}"
        })
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    
    # Summary statistics
    total_tokens = sum(exp['total_tokens'] for exp in experiments)
    total_cost = sum(exp['total_input_tokens'] * 0.00001875 + exp['total_output_tokens'] * 0.000075 for exp in experiments)
    total_generations = sum(exp['total_generations'] for exp in experiments)
    
    print(f"\n📊 OVERALL STATISTICS")
    print(f"Total Experiments: {len(experiments)}")
    print(f"Total Generations: {total_generations:,}")
    print(f"Total Tokens Used: {total_tokens:,}")
    print(f"Total Estimated Cost: ${total_cost:.4f}")


def show_experiment_details(experiment_id: str, log_dir: str = ".") -> None:
    """Show detailed information for a specific experiment"""
    log_file = f"{log_dir}/{experiment_id}_log.json"
    
    try:
        with open(log_file, 'r') as f:
            exp = json.load(f)
    except FileNotFoundError:
        print(f"Experiment log not found: {log_file}")
        return
    
    print(f"🧪 EXPERIMENT DETAILS: {experiment_id}")
    print("=" * 50)
    
    # Show configuration
    print("⚙️  Configuration:")
    for key, value in exp['config'].items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                print(f"    {subkey}: {subvalue}")
        else:
            print(f"  {key}: {value}")
    
    # Show metrics
    print(f"\n📊 Metrics:")
    print(f"  Total Duration: {exp.get('total_duration_seconds', 0) / 60:.1f} minutes")
    print(f"  Total Generations: {exp['total_generations']}")
    print(f"  Total Tokens: {exp['total_tokens']:,}")
    print(f"  Input Tokens: {exp['total_input_tokens']:,}")
    print(f"  Output Tokens: {exp['total_output_tokens']:,}")
    
    # Show per-generation breakdown
    if exp['metrics']:
        print(f"\n🔍 Generation Breakdown:")
        for i, metric in enumerate(exp['metrics'][:5]):  # Show first 5
            print(f"  Frame {metric['frame_number']}: {metric['duration_seconds']:.2f}s, "
                  f"{metric['token_usage']['total_tokens']:,} tokens")
        
        if len(exp['metrics']) > 5:
            print(f"  ... and {len(exp['metrics']) - 5} more generations")


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment logs")
    parser.add_argument("--dir", default=".", help="Directory containing log files")
    parser.add_argument("--experiment", help="Show details for specific experiment ID")
    parser.add_argument("--list", action="store_true", help="List all experiments")
    
    args = parser.parse_args()
    
    if args.experiment:
        show_experiment_details(args.experiment, args.dir)
    else:
        experiments = load_experiment_logs(args.dir)
        analyze_experiments(experiments)


if __name__ == "__main__":
    try:
        import pandas as pd
    except ImportError:
        print("Warning: pandas not installed. Install with 'pip install pandas' for better formatting.")
        # Fallback without pandas
        pd = None
    
    main()
