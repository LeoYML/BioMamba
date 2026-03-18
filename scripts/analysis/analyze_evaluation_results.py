"""
Analyze and compare evaluation results from multiple models
"""

# --- Project root resolution (auto-generated) ---
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')
_sys.path.insert(0, _PROJECT_ROOT)
_os.chdir(_PROJECT_ROOT)
# --- End project root resolution ---

import os
import json
import argparse
import glob
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze evaluation results')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing evaluation results')
    parser.add_argument('--output_file', type=str, default='comparison_report.txt',
                        help='Output file for comparison report')
    parser.add_argument('--create_plots', action='store_true',
                        help='Create visualization plots')
    return parser.parse_args()


def load_results(results_dir: str) -> List[Dict]:
    """Load all metrics files from results directory"""
    results = []
    
    # Find all metrics files
    metrics_files = glob.glob(os.path.join(results_dir, '**/*_metrics.json'), recursive=True)
    
    print(f"Found {len(metrics_files)} metrics files in {results_dir}")
    
    for file_path in metrics_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                results.append({
                    'file': file_path,
                    'model_path': data.get('model_path', 'unknown'),
                    'model_type': data.get('model_type', 'unknown'),
                    'dataset': data.get('dataset', 'unknown'),
                    'metrics': data.get('metrics', {})
                })
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return results


def create_comparison_table(results: List[Dict]) -> pd.DataFrame:
    """Create comparison table from results"""
    rows = []
    
    for result in results:
        model_name = os.path.basename(result['model_path'])
        metrics = result['metrics']
        
        row = {
            'Model': model_name,
            'Type': result['model_type'],
            'Dataset': result['dataset'],
            'Accuracy': metrics.get('accuracy', 0),
            'F1': metrics.get('f1', 0),
            'Precision': metrics.get('precision', 0),
            'Recall': metrics.get('recall', 0),
            'Samples': metrics.get('valid_samples', 0)
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def print_report(df: pd.DataFrame, output_file: str):
    """Print comparison report"""
    report = []
    
    report.append("="*80)
    report.append("EVALUATION RESULTS COMPARISON")
    report.append("="*80)
    report.append("")
    
    # Summary statistics
    report.append("SUMMARY")
    report.append("-"*80)
    report.append(f"Total models evaluated: {len(df)}")
    report.append(f"Datasets: {', '.join(df['Dataset'].unique())}")
    report.append(f"Model types: {', '.join(df['Type'].unique())}")
    report.append("")
    
    # Rankings
    report.append("RANKINGS BY ACCURACY")
    report.append("-"*80)
    df_sorted = df.sort_values('Accuracy', ascending=False)
    for idx, row in df_sorted.iterrows():
        report.append(f"{idx+1}. {row['Model']} ({row['Type']}): {row['Accuracy']:.4f} ({row['Accuracy']*100:.2f}%)")
    report.append("")
    
    report.append("RANKINGS BY F1 SCORE")
    report.append("-"*80)
    df_sorted = df.sort_values('F1', ascending=False)
    for idx, row in df_sorted.iterrows():
        report.append(f"{idx+1}. {row['Model']} ({row['Type']}): {row['F1']:.4f}")
    report.append("")
    
    # Detailed comparison table
    report.append("DETAILED COMPARISON")
    report.append("-"*80)
    report.append(df.to_string(index=False))
    report.append("")
    
    # Best model
    best_acc = df.loc[df['Accuracy'].idxmax()]
    best_f1 = df.loc[df['F1'].idxmax()]
    
    report.append("BEST MODELS")
    report.append("-"*80)
    report.append(f"Best Accuracy: {best_acc['Model']} ({best_acc['Type']}) - {best_acc['Accuracy']:.4f}")
    report.append(f"Best F1 Score: {best_f1['Model']} ({best_f1['Type']}) - {best_f1['F1']:.4f}")
    report.append("")
    
    report.append("="*80)
    
    # Print to console
    report_text = '\n'.join(report)
    print(report_text)
    
    # Save to file
    with open(output_file, 'w') as f:
        f.write(report_text)
    
    print(f"\nReport saved to: {output_file}")


def create_plots(df: pd.DataFrame, output_dir: str):
    """Create visualization plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Accuracy comparison bar plot
    plt.figure(figsize=(12, 6))
    df_sorted = df.sort_values('Accuracy', ascending=True)
    colors = sns.color_palette("husl", len(df))
    plt.barh(range(len(df_sorted)), df_sorted['Accuracy'], color=colors)
    plt.yticks(range(len(df_sorted)), df_sorted['Model'])
    plt.xlabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300)
    print(f"Saved: {os.path.join(output_dir, 'accuracy_comparison.png')}")
    plt.close()
    
    # 2. Multi-metric comparison
    plt.figure(figsize=(12, 8))
    metrics = ['Accuracy', 'F1', 'Precision', 'Recall']
    x = range(len(df))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        plt.bar([j + i*width for j in x], df[metric], width, label=metric)
    
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Multi-Metric Comparison')
    plt.xticks([j + width*1.5 for j in x], df['Model'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multi_metric_comparison.png'), dpi=300)
    print(f"Saved: {os.path.join(output_dir, 'multi_metric_comparison.png')}")
    plt.close()
    
    # 3. Radar chart for best model
    if len(df) > 0:
        best_model = df.loc[df['Accuracy'].idxmax()]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        categories = ['Accuracy', 'F1', 'Precision', 'Recall']
        values = [best_model[cat] for cat in categories]
        values += values[:1]  # Complete the circle
        
        angles = [n / float(len(categories)) * 2 * 3.14159 for n in range(len(categories))]
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        plt.title(f"Best Model Performance: {best_model['Model']}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'best_model_radar.png'), dpi=300)
        print(f"Saved: {os.path.join(output_dir, 'best_model_radar.png')}")
        plt.close()


def main():
    args = parse_args()
    
    print("="*80)
    print("EVALUATION RESULTS ANALYZER")
    print("="*80)
    print("")
    
    # Load results
    print(f"Loading results from: {args.results_dir}")
    results = load_results(args.results_dir)
    
    if not results:
        print("Error: No results found!")
        return
    
    print(f"Loaded {len(results)} result files")
    print("")
    
    # Create comparison table
    df = create_comparison_table(results)
    
    # Print report
    print_report(df, args.output_file)
    
    # Create plots if requested
    if args.create_plots:
        print("\nCreating visualization plots...")
        plots_dir = os.path.join(os.path.dirname(args.output_file), 'plots')
        try:
            create_plots(df, plots_dir)
            print(f"Plots saved to: {plots_dir}")
        except Exception as e:
            print(f"Warning: Could not create plots: {e}")
            print("Install matplotlib and seaborn to enable plotting:")
            print("  pip install matplotlib seaborn")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
