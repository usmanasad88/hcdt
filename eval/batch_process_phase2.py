#!/usr/bin/env python3
"""
Batch processor for multiple Phase 2 experiment results.
Useful for comparing different model outputs or experiment runs.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eval.visualize_phase_two import Phase2Visualizer


class Phase2BatchProcessor:
    """Batch processor for multiple phase2 results."""
    
    def __init__(self):
        self.visualizer = Phase2Visualizer()
        self.results = {}
        
    def find_all_phase2_results(self, directory: str = "logs") -> List[str]:
        """Find all phase2 results files in a directory."""
        if not os.path.exists(directory):
            return []
            
        phase2_files = []
        for file in os.listdir(directory):
            if 'phase2' in file.lower() and file.endswith('.json'):
                phase2_files.append(os.path.join(directory, file))
        
        return sorted(phase2_files)
    
    def process_single_file(self, results_file: str, output_suffix: str = "") -> Dict:
        """Process a single results file and return summary statistics."""
        print(f"\n📂 Processing: {os.path.basename(results_file)}")
        
        # Load results
        if not self.visualizer.load_results(results_file):
            return None
        
        # Create output directory with suffix
        base_name = f"phase2_batch_{os.path.basename(results_file).replace('.json', '')}{output_suffix}"
        output_dir = self.visualizer.create_output_directory(base_name)
        
        # Run visualization (quick mode for batch processing)
        self.visualizer.visualize_predictions_on_frames(show_ground_truth=True)
        errors = self.visualizer.calculate_prediction_errors()
        
        # Extract key statistics
        summary = {
            'file': results_file,
            'output_dir': output_dir,
            'total_predictions': len(self.visualizer.results_data),
            'errors': errors
        }
        
        if errors and 'statistics' in errors:
            for hand_type, stats in errors['statistics'].items():
                summary[f'{hand_type}_mean'] = stats['mean']
                summary[f'{hand_type}_std'] = stats['std']
                summary[f'{hand_type}_median'] = stats['median']
        
        return summary
    
    def create_comparison_report(self, all_results: List[Dict], output_dir: str):
        """Create a comparison report across all processed files."""
        comparison_file = os.path.join(output_dir, "batch_comparison_report.txt")
        
        with open(comparison_file, 'w') as f:
            f.write("Phase 2 Batch Processing Comparison Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Total files processed: {len(all_results)}\n\n")
            
            # Summary table
            f.write("Summary Statistics:\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'File':<30} {'Predictions':<12} {'Left Err':<10} {'Right Err':<10} {'Combined':<10}\n")
            f.write("-" * 80 + "\n")
            
            for result in all_results:
                if result:
                    filename = os.path.basename(result['file'])[:25]
                    total_pred = result['total_predictions']
                    left_err = result.get('left_hand_errors_mean', 0)
                    right_err = result.get('right_hand_errors_mean', 0)
                    combined_err = result.get('combined_errors_mean', 0)
                    
                    f.write(f"{filename:<30} {total_pred:<12} {left_err:<10.2f} {right_err:<10.2f} {combined_err:<10.2f}\n")
            
            f.write("\n\nDetailed Statistics:\n")
            f.write("-" * 30 + "\n")
            
            for result in all_results:
                if result:
                    f.write(f"\nFile: {os.path.basename(result['file'])}\n")
                    f.write(f"  Total Predictions: {result['total_predictions']}\n")
                    f.write(f"  Output Directory: {result['output_dir']}\n")
                    
                    if 'errors' in result and result['errors']:
                        errors = result['errors']
                        if 'statistics' in errors:
                            for hand_type, stats in errors['statistics'].items():
                                f.write(f"  {hand_type.replace('_', ' ').title()}:\n")
                                f.write(f"    Mean: {stats['mean']:.2f}\n")
                                f.write(f"    Std:  {stats['std']:.2f}\n")
                                f.write(f"    Range: {stats['min']:.2f} - {stats['max']:.2f}\n")
        
        print(f"📊 Comparison report saved: {comparison_file}")
        
        # Also create a CSV for easy analysis
        csv_data = []
        for result in all_results:
            if result:
                row = {
                    'filename': os.path.basename(result['file']),
                    'total_predictions': result['total_predictions']
                }
                
                # Add error statistics
                for key, value in result.items():
                    if '_mean' in key or '_std' in key or '_median' in key:
                        row[key] = value
                
                csv_data.append(row)
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_file = os.path.join(output_dir, "batch_comparison_data.csv")
            df.to_csv(csv_file, index=False)
            print(f"📈 Comparison data CSV saved: {csv_file}")
    
    def process_batch(self, results_files: List[str] = None, directory: str = "logs") -> str:
        """Process multiple results files in batch."""
        if not results_files:
            results_files = self.find_all_phase2_results(directory)
        
        if not results_files:
            print("❌ No phase2 results files found.")
            return None
        
        print(f"🚀 Starting batch processing of {len(results_files)} files...")
        
        # Create overall output directory
        timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_output_dir = os.path.join("outputs", f"phase2_batch_{timestamp}")
        os.makedirs(batch_output_dir, exist_ok=True)
        
        all_results = []
        
        for i, results_file in enumerate(results_files, 1):
            try:
                suffix = f"_{i:02d}"
                result = self.process_single_file(results_file, suffix)
                if result:
                    all_results.append(result)
                    print(f"✅ Processed {i}/{len(results_files)}: {os.path.basename(results_file)}")
                else:
                    print(f"❌ Failed to process: {os.path.basename(results_file)}")
            except Exception as e:
                print(f"❌ Error processing {results_file}: {e}")
        
        # Create comparison report
        if all_results:
            self.create_comparison_report(all_results, batch_output_dir)
        
        print(f"\n🎉 Batch processing completed!")
        print(f"📁 Results saved to: {batch_output_dir}")
        print(f"✅ Successfully processed: {len(all_results)}/{len(results_files)} files")
        
        return batch_output_dir


def main():
    parser = argparse.ArgumentParser(description="Batch process Phase 2 results")
    parser.add_argument("--directory", "-d", default="logs", 
                       help="Directory to search for phase2 results (default: logs)")
    parser.add_argument("--files", "-f", nargs="+", 
                       help="Specific files to process (overrides directory search)")
    parser.add_argument("--list", action="store_true", 
                       help="List available phase2 results files")
    
    args = parser.parse_args()
    
    processor = Phase2BatchProcessor()
    
    if args.list:
        files = processor.find_all_phase2_results(args.directory)
        if files:
            print(f"Found {len(files)} phase2 results files in '{args.directory}':")
            for f in files:
                mtime = os.path.getmtime(f)
                mtime_str = __import__('datetime').datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                print(f"  - {os.path.basename(f)} ({mtime_str})")
        else:
            print(f"No phase2 results files found in '{args.directory}'")
        return
    
    # Process files
    result_dir = processor.process_batch(args.files, args.directory)
    
    if result_dir:
        print(f"\n✅ Batch processing completed successfully!")
        print(f"📁 All results available at: {result_dir}")
    else:
        print("\n❌ Batch processing failed.")


if __name__ == "__main__":
    main()
