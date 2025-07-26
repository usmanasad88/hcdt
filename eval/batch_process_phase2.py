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
import matplotlib.pyplot as plt
import numpy as np

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
        
        # Generate comparison plots
        self.create_comparison_plots(all_results, output_dir)
        
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
    
    def create_comparison_plots(self, all_results: List[Dict], output_dir: str):
        """Create comprehensive comparison plots across all processed files."""
        # Collect error data for all files
        left_hand_data = []
        right_hand_data = []
        combined_data = []
        file_labels = []
        mean_errors = {'left': [], 'right': [], 'combined': []}
        
        for result in all_results:
            if result and 'errors' in result and result['errors']:
                errors = result['errors']
                filename = os.path.basename(result['file']).replace('.json', '')
                
                # Collect raw error arrays for box plots
                if 'left_hand_errors' in errors and errors['left_hand_errors']:
                    left_hand_data.append(errors['left_hand_errors'])
                    file_labels.append(filename)
                
                if 'right_hand_errors' in errors and errors['right_hand_errors']:
                    right_hand_data.append(errors['right_hand_errors'])
                
                if 'combined_errors' in errors and errors['combined_errors']:
                    combined_data.append(errors['combined_errors'])
                
                # Collect mean errors for bar charts
                if 'statistics' in errors:
                    stats = errors['statistics']
                    mean_errors['left'].append(stats.get('left_hand_errors', {}).get('mean', 0))
                    mean_errors['right'].append(stats.get('right_hand_errors', {}).get('mean', 0))
                    mean_errors['combined'].append(stats.get('combined_errors', {}).get('mean', 0))
        
        if not file_labels:
            print("⚠️  No error data available for comparison plots")
            return
        
        # Create comprehensive comparison plots
        self._create_box_plot_comparison(left_hand_data, right_hand_data, combined_data, file_labels, output_dir)
        self._create_statistical_comparison(all_results, output_dir)
        self._create_performance_ranking(all_results, output_dir)
        
    def _create_box_plot_comparison(self, left_hand_data, right_hand_data, combined_data, file_labels, output_dir):
        """Create box plots comparing error distributions across all files."""
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('Batch Processing Results Comparison - Box Plots', fontsize=16, fontweight='bold')
        
        # Prepare labels (truncate if too long)
        display_labels = [label[:15] + '...' if len(label) > 15 else label for label in file_labels]
        
        # Left hand errors box plot
        if left_hand_data:
            bp1 = axes[0, 0].boxplot(left_hand_data, tick_labels=display_labels, patch_artist=True)
            axes[0, 0].set_title('Left Hand Prediction Errors Comparison', fontweight='bold')
            axes[0, 0].set_ylabel('Error (normalized pixels)')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
            
            # Color the boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(bp1['boxes'])))
            for patch, color in zip(bp1['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        # Right hand errors box plot
        if right_hand_data and len(right_hand_data) == len(display_labels):
            bp2 = axes[0, 1].boxplot(right_hand_data, tick_labels=display_labels, patch_artist=True)
            axes[0, 1].set_title('Right Hand Prediction Errors Comparison', fontweight='bold')
            axes[0, 1].set_ylabel('Error (normalized pixels)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
            
            # Color the boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(bp2['boxes'])))
            for patch, color in zip(bp2['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        # Combined errors box plot
        if combined_data and len(combined_data) == len(display_labels):
            bp3 = axes[1, 0].boxplot(combined_data, tick_labels=display_labels, patch_artist=True)
            axes[1, 0].set_title('Combined Hand Prediction Errors Comparison', fontweight='bold')
            axes[1, 0].set_ylabel('Error (normalized pixels)')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
            
            # Color the boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(bp3['boxes'])))
            for patch, color in zip(bp3['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        # Summary violin plot for combined errors
        if combined_data and len(combined_data) == len(display_labels):
            parts = axes[1, 1].violinplot(combined_data, positions=range(1, len(combined_data) + 1), 
                                         showmeans=True, showmedians=True)
            axes[1, 1].set_title('Combined Error Distribution Density', fontweight='bold')
            axes[1, 1].set_ylabel('Error (normalized pixels)')
            axes[1, 1].set_xticks(range(1, len(display_labels) + 1))
            axes[1, 1].set_xticklabels(display_labels, rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
            
            # Color the violin plots
            colors = plt.cm.viridis(np.linspace(0, 1, len(parts['bodies'])))
            for pc, color in zip(parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
        
        plt.tight_layout()
        
        # Save the box plot comparison
        box_plot_path = os.path.join(output_dir, "batch_comparison_box_plots.png")
        plt.savefig(box_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Batch comparison box plots saved: {box_plot_path}")
    
    def _create_statistical_comparison(self, all_results, output_dir):
        """Create detailed statistical comparison plots."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Detailed Statistical Comparison Across Batch Results', fontsize=14, fontweight='bold')
        
        # Collect statistics
        filenames = []
        left_stats = {'mean': [], 'std': [], 'median': [], 'min': [], 'max': []}
        right_stats = {'mean': [], 'std': [], 'median': [], 'min': [], 'max': []}
        combined_stats = {'mean': [], 'std': [], 'median': [], 'min': [], 'max': []}
        
        for result in all_results:
            if result and 'errors' in result and result['errors'] and 'statistics' in result['errors']:
                stats = result['errors']['statistics']
                filename = os.path.basename(result['file']).replace('.json', '')
                filenames.append(filename[:12] + '...' if len(filename) > 12 else filename)
                
                # Left hand stats
                left_stat = stats.get('left_hand_errors', {})
                for key in left_stats:
                    left_stats[key].append(left_stat.get(key, 0))
                
                # Right hand stats
                right_stat = stats.get('right_hand_errors', {})
                for key in right_stats:
                    right_stats[key].append(right_stat.get(key, 0))
                
                # Combined stats
                combined_stat = stats.get('combined_errors', {})
                for key in combined_stats:
                    combined_stats[key].append(combined_stat.get(key, 0))
        
        if not filenames:
            return
        
        x_pos = range(len(filenames))
        
        # Mean comparison
        axes[0, 0].plot(x_pos, left_stats['mean'], 'ro-', label='Left Hand', linewidth=2, markersize=6)
        axes[0, 0].plot(x_pos, right_stats['mean'], 'bo-', label='Right Hand', linewidth=2, markersize=6)
        axes[0, 0].plot(x_pos, combined_stats['mean'], 'go-', label='Combined', linewidth=2, markersize=6)
        axes[0, 0].set_title('Mean Error Comparison')
        axes[0, 0].set_ylabel('Mean Error (normalized pixels)')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(filenames, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Standard deviation comparison
        axes[0, 1].plot(x_pos, left_stats['std'], 'r^-', label='Left Hand', linewidth=2, markersize=6)
        axes[0, 1].plot(x_pos, right_stats['std'], 'b^-', label='Right Hand', linewidth=2, markersize=6)
        axes[0, 1].plot(x_pos, combined_stats['std'], 'g^-', label='Combined', linewidth=2, markersize=6)
        axes[0, 1].set_title('Standard Deviation Comparison')
        axes[0, 1].set_ylabel('Standard Deviation')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(filenames, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Error range visualization
        axes[1, 0].fill_between(x_pos, left_stats['min'], left_stats['max'], 
                               alpha=0.3, color='red', label='Left Hand Range')
        axes[1, 0].fill_between(x_pos, right_stats['min'], right_stats['max'], 
                               alpha=0.3, color='blue', label='Right Hand Range')
        axes[1, 0].plot(x_pos, left_stats['median'], 'ro-', label='Left Median', markersize=4)
        axes[1, 0].plot(x_pos, right_stats['median'], 'bo-', label='Right Median', markersize=4)
        axes[1, 0].set_title('Error Range and Median Comparison')
        axes[1, 0].set_ylabel('Error Value (normalized pixels)')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(filenames, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Combined error bar chart with error bars
        mean_combined = combined_stats['mean']
        std_combined = combined_stats['std']
        
        bars = axes[1, 1].bar(x_pos, mean_combined, yerr=std_combined, 
                             capsize=5, alpha=0.7, color='skyblue', 
                             edgecolor='navy', linewidth=1)
        axes[1, 1].set_title('Combined Error with Standard Deviation')
        axes[1, 1].set_ylabel('Mean Error ± Std Dev')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(filenames, rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Color bars based on performance (green=good, red=poor)
        max_error = max(mean_combined) if mean_combined else 1
        for bar, error in zip(bars, mean_combined):
            # Normalize to 0-1 and use for coloring
            normalized_error = error / max_error
            if normalized_error < 0.33:
                bar.set_color('lightgreen')
            elif normalized_error < 0.66:
                bar.set_color('orange')
            else:
                bar.set_color('lightcoral')
        
        plt.tight_layout()
        
        # Save the detailed statistical plot
        detailed_plot_path = os.path.join(output_dir, "batch_detailed_statistical_comparison.png")
        plt.savefig(detailed_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Detailed statistical comparison plot saved: {detailed_plot_path}")
    
    def _create_performance_ranking(self, all_results, output_dir):
        """Create performance ranking visualization."""
        # Extract performance data
        performance_data = []
        for result in all_results:
            if result and 'errors' in result and result['errors'] and 'statistics' in result['errors']:
                filename = os.path.basename(result['file']).replace('.json', '')
                stats = result['errors']['statistics']
                combined_mean = stats.get('combined_errors', {}).get('mean', float('inf'))
                combined_std = stats.get('combined_errors', {}).get('std', 0)
                total_predictions = result['total_predictions']
                
                performance_data.append({
                    'filename': filename,
                    'combined_mean': combined_mean,
                    'combined_std': combined_std,
                    'total_predictions': total_predictions,
                    'score': combined_mean  # Lower is better
                })
        
        if not performance_data:
            return
        
        # Sort by performance (lower error is better)
        performance_data.sort(key=lambda x: x['score'])
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Performance Ranking and Analysis', fontsize=14, fontweight='bold')
        
        # Performance ranking bar chart
        filenames = [item['filename'][:15] + '...' if len(item['filename']) > 15 else item['filename'] 
                    for item in performance_data]
        scores = [item['score'] for item in performance_data]
        
        # Color gradient from best (green) to worst (red)
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(scores)))
        
        bars = axes[0].bar(range(len(filenames)), scores, color=colors, alpha=0.8, edgecolor='black')
        axes[0].set_title('Performance Ranking (Lower Error = Better)')
        axes[0].set_ylabel('Combined Mean Error')
        axes[0].set_xticks(range(len(filenames)))
        axes[0].set_xticklabels(filenames, rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + max(scores)*0.01,
                        f'{score:.2f}', ha='center', va='bottom', fontsize=9)
        
        # Scatter plot: Error vs Number of Predictions
        x_vals = [item['total_predictions'] for item in performance_data]
        y_vals = [item['combined_mean'] for item in performance_data]
        
        scatter = axes[1].scatter(x_vals, y_vals, c=scores, cmap='RdYlGn_r', 
                                 s=100, alpha=0.7, edgecolors='black')
        axes[1].set_title('Error vs Number of Predictions')
        axes[1].set_xlabel('Total Predictions')
        axes[1].set_ylabel('Combined Mean Error')
        axes[1].grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=axes[1])
        cbar.set_label('Combined Mean Error')
        
        # Add filename labels to points
        for i, item in enumerate(performance_data):
            short_name = item['filename'][:8] + '...' if len(item['filename']) > 8 else item['filename']
            axes[1].annotate(short_name, (x_vals[i], y_vals[i]), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, alpha=0.8)
        
        plt.tight_layout()
        
        # Save performance ranking plot
        ranking_plot_path = os.path.join(output_dir, "batch_performance_ranking.png")
        plt.savefig(ranking_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"🏆 Performance ranking plot saved: {ranking_plot_path}")
        
        # Save ranking to text file
        ranking_file = os.path.join(output_dir, "performance_ranking.txt")
        with open(ranking_file, 'w') as f:
            f.write("Performance Ranking (Best to Worst)\n")
            f.write("=" * 50 + "\n\n")
            
            for i, item in enumerate(performance_data, 1):
                f.write(f"{i:2d}. {item['filename']}\n")
                f.write(f"    Combined Mean Error: {item['combined_mean']:.2f}\n")
                f.write(f"    Standard Deviation: {item['combined_std']:.2f}\n")
                f.write(f"    Total Predictions: {item['total_predictions']}\n\n")
        
        print(f"📋 Performance ranking saved: {ranking_file}")
    
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
    parser.add_argument("--directory", "-d", default="logs/Phase2_results", 
                       help="Directory to search for phase2 results (default: logs/Phase2_results)")
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
