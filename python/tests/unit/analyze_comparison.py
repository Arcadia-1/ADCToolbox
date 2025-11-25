"""analyze_comparison.py - Analyze MATLAB vs Python consistency

⚠️ DEPRECATION WARNING ⚠️
==========================
This script is DEPRECATED and will be removed in a future version.

Please use the unified comparison system instead:
    python tests/system/universal_csv_compare.py
    python tests/system/compare_all.py

The new comparison system provides comprehensive analysis with better
reporting, filtering options, and integration with the test hierarchy.

This file is kept temporarily for backwards compatibility only.
==========================

Analyzes all comparison.csv files and generates a summary report.
"""

import pandas as pd
import os
from pathlib import Path
from glob import glob


def main():
    """Analyze all comparison files."""
    test_output_dir = "test_output"

    # Find all comparison files
    comparison_files = glob(os.path.join(test_output_dir, "**", "test_specPlot", "comparison.csv"), recursive=True)

    if not comparison_files:
        print("No comparison files found.")
        return

    print("=" * 80)
    print("MATLAB vs Python Consistency Analysis - test_specPlot")
    print("=" * 80)
    print()

    # Storage for results
    results = []

    for comp_file in sorted(comparison_files):
        # Extract dataset name from path
        path_parts = Path(comp_file).parts
        dataset_name = path_parts[-3]  # test_output/<dataset_name>/test_specPlot/comparison.csv

        # Read comparison
        df = pd.read_csv(comp_file)

        # Calculate summary statistics
        max_diff = df['Diff'].abs().max()
        max_diff_pct = df['Diff_pct'].abs().max()
        metric_with_max_diff = df.loc[df['Diff_pct'].abs().idxmax(), 'Metric']

        # Determine status
        if max_diff_pct < 0.1:
            status = "EXCELLENT"
        elif max_diff_pct < 1.0:
            status = "GOOD"
        elif max_diff_pct < 5.0:
            status = "ACCEPTABLE"
        else:
            status = "NEEDS REVIEW"

        results.append({
            'Dataset': dataset_name,
            'Max_Diff': max_diff,
            'Max_Diff_Pct': max_diff_pct,
            'Worst_Metric': metric_with_max_diff,
            'Status': status
        })

    # Create summary dataframe
    summary_df = pd.DataFrame(results)

    # Count by status
    status_counts = summary_df['Status'].value_counts()

    print(f"Total datasets tested: {len(results)}")
    print()
    print("Status Summary:")
    for status in ['EXCELLENT', 'GOOD', 'ACCEPTABLE', 'NEEDS REVIEW']:
        count = status_counts.get(status, 0)
        pct = 100 * count / len(results) if len(results) > 0 else 0
        print(f"  {status:15s}: {count:3d} ({pct:5.1f}%)")
    print()

    # Show best matches
    print("=" * 80)
    print("Top 10 Best Matches (Lowest % Difference)")
    print("=" * 80)
    best_matches = summary_df.nsmallest(10, 'Max_Diff_Pct')
    for idx, row in best_matches.iterrows():
        print(f"{row['Dataset']:50s} - {row['Max_Diff_Pct']:7.3f}% diff in {row['Worst_Metric']}")
    print()

    # Show worst matches
    print("=" * 80)
    print("Top 10 Datasets Needing Review (Highest % Difference)")
    print("=" * 80)
    worst_matches = summary_df.nlargest(10, 'Max_Diff_Pct')
    for idx, row in worst_matches.iterrows():
        print(f"{row['Dataset']:50s} - {row['Max_Diff_Pct']:7.2f}% diff in {row['Worst_Metric']}")
    print()

    # Show detailed comparison for worst case
    print("=" * 80)
    print("Detailed Comparison for Worst Case Dataset")
    print("=" * 80)
    worst_idx = summary_df['Max_Diff_Pct'].idxmax()
    worst_dataset = summary_df.loc[worst_idx, 'Dataset']
    worst_file = os.path.join(test_output_dir, worst_dataset, "test_specPlot", "comparison.csv")

    print(f"Dataset: {worst_dataset}")
    print()
    worst_df = pd.read_csv(worst_file)
    pd.options.display.float_format = '{:.4f}'.format
    print(worst_df.to_string(index=False))
    print()

    # Save summary to CSV
    summary_path = os.path.join(test_output_dir, "test_specPlot_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")
    print()

    # Analysis insights
    print("=" * 80)
    print("Analysis Insights")
    print("=" * 80)

    # Check which types of datasets have issues
    needs_review = summary_df[summary_df['Status'] == 'NEEDS REVIEW']

    if len(needs_review) > 0:
        print("\nDatasets needing review:")
        for pattern in ['HD', 'clipping', 'drift', 'amplitude_modulation', 'gain_error', 'kickback', 'ref_error']:
            matching = needs_review[needs_review['Dataset'].str.contains(pattern, case=False)]
            if len(matching) > 0:
                print(f"  - {pattern}: {len(matching)} dataset(s)")

        print("\nCommon issues:")
        worst_metrics = needs_review['Worst_Metric'].value_counts()
        for metric, count in worst_metrics.items():
            print(f"  - {metric}: {count} dataset(s)")
    else:
        print("\nAll datasets show good consistency!")

    print()


if __name__ == "__main__":
    main()
