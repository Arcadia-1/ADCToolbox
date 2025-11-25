"""compare_all_csv_pairs.py - Compare all MATLAB vs Python CSV pairs

⚠️ DEPRECATION WARNING ⚠️
==========================
This script is DEPRECATED and will be removed in a future version.

Please use the unified comparison system instead:
    python tests/system/universal_csv_compare.py

The universal_csv_compare.py tool provides all the same functionality
plus additional features:
- Phase wrapping for phase angle comparisons
- Machine precision handling (1e-10 tolerance)
- Zero-magnitude phase handling
- Better error reporting and classification
- Integration with the hierarchical comparison system

This file is kept temporarily for backwards compatibility only.
==========================

Auto-discovers all *_python.csv and *_matlab.csv pairs in test_output/ directory
and generates a comprehensive comparison report.

Usage (DEPRECATED):
    cd d:\ADCToolbox
    python tests/unit/compare_all_csv_pairs.py
"""

import pandas as pd
from pathlib import Path
import numpy as np

#============================================================================
# SETTINGS (modify directly)
#============================================================================

# Filter by test type (leave empty to analyze all)
TEST_TYPES = []  # e.g., ['test_specPlot', 'test_sineFit'] or [] for all

# Filter by dataset (leave empty to analyze all)
DATASETS = []  # e.g., ['sinewave_clipping_0P055'] or [] for all

# Show only problems (>1% difference)
PROBLEMS_ONLY = False  # Set to True to hide good results

# Difference thresholds for classification
THRESHOLD_EXCELLENT = 0.1    # < 0.1% diff
THRESHOLD_GOOD = 1.0         # < 1.0% diff
THRESHOLD_ACCEPTABLE = 5.0   # < 5.0% diff
# Above 5.0% = NEEDS REVIEW

# Output directory
TEST_OUTPUT_DIR = 'test_output'

#============================================================================


def find_csv_pairs(output_dir):
    """Find all *_python.csv and *_matlab.csv pairs."""
    output_path = Path(output_dir)

    if not output_path.exists():
        print(f"[ERROR] Output directory not found: {output_dir}")
        return []

    # Find all python CSV files
    python_files = list(output_path.glob('**/*_python.csv'))

    pairs = []
    for python_file in python_files:
        # Construct corresponding matlab file path
        matlab_file = python_file.parent / python_file.name.replace('_python.csv', '_matlab.csv')

        if matlab_file.exists():
            # Extract metadata from path
            parts = python_file.relative_to(output_path).parts
            dataset_name = parts[0] if len(parts) > 0 else 'unknown'
            test_type = parts[1] if len(parts) > 1 else 'unknown'
            csv_name = python_file.stem.replace('_python', '')

            pairs.append({
                'python_file': python_file,
                'matlab_file': matlab_file,
                'dataset': dataset_name,
                'test_type': test_type,
                'csv_name': csv_name,
                'pair_name': f"{dataset_name}/{test_type}/{csv_name}"
            })

    return pairs


def compare_csv_pair(python_file, matlab_file):
    """Compare two CSV files and return difference statistics."""
    try:
        df_py = pd.read_csv(python_file)
        df_mat = pd.read_csv(matlab_file)

        # Check if this is a metrics-style CSV (has 'Metric' column)
        if 'Metric' in df_py.columns and 'MATLAB' in df_py.columns and 'Python' in df_py.columns:
            return compare_metrics_csv(df_py, df_mat)
        else:
            return compare_numeric_csv(df_py, df_mat)

    except Exception as e:
        return {
            'status': 'ERROR',
            'max_diff_pct': np.nan,
            'max_diff_abs': np.nan,
            'worst_metric': 'N/A',
            'error_msg': str(e)
        }


def compare_metrics_csv(df_py, df_mat):
    """Compare metrics-style CSV (like test_specPlot output)."""
    # Merge on Metric column
    if 'Metric' not in df_py.columns or 'Metric' not in df_mat.columns:
        return compare_numeric_csv(df_py, df_mat)

    merged = df_py.merge(df_mat, on='Metric', suffixes=('_py', '_mat'))

    # Calculate differences
    if 'Python' in df_py.columns and 'MATLAB' in df_mat.columns:
        merged['Diff'] = merged['Python'] - merged['MATLAB']
        merged['Diff_pct'] = np.abs(merged['Diff'] / merged['MATLAB'] * 100)
        merged['Diff_pct'] = merged['Diff_pct'].replace([np.inf, -np.inf], np.nan)

        max_diff_pct = merged['Diff_pct'].max()
        max_diff_abs = merged['Diff'].abs().max()
        worst_metric = merged.loc[merged['Diff_pct'].idxmax(), 'Metric'] if not merged['Diff_pct'].isna().all() else 'N/A'
    else:
        # Fallback to numeric comparison
        return compare_numeric_csv(df_py, df_mat)

    # Classify status
    if np.isnan(max_diff_pct):
        status = 'ERROR'
    elif max_diff_pct < THRESHOLD_EXCELLENT:
        status = 'EXCELLENT'
    elif max_diff_pct < THRESHOLD_GOOD:
        status = 'GOOD'
    elif max_diff_pct < THRESHOLD_ACCEPTABLE:
        status = 'ACCEPTABLE'
    else:
        status = 'NEEDS REVIEW'

    return {
        'status': status,
        'max_diff_pct': max_diff_pct,
        'max_diff_abs': max_diff_abs,
        'worst_metric': worst_metric,
        'error_msg': None
    }


def compare_numeric_csv(df_py, df_mat):
    """Compare numeric CSV files (arrays, time series, etc.)."""
    try:
        # Convert to numeric arrays
        py_values = df_py.select_dtypes(include=[np.number]).values.flatten()
        mat_values = df_mat.select_dtypes(include=[np.number]).values.flatten()

        if len(py_values) == 0 or len(mat_values) == 0:
            return {
                'status': 'ERROR',
                'max_diff_pct': np.nan,
                'max_diff_abs': np.nan,
                'worst_metric': 'N/A',
                'error_msg': 'No numeric data found'
            }

        if len(py_values) != len(mat_values):
            return {
                'status': 'ERROR',
                'max_diff_pct': np.nan,
                'max_diff_abs': np.nan,
                'worst_metric': 'N/A',
                'error_msg': f'Length mismatch: Python={len(py_values)}, MATLAB={len(mat_values)}'
            }

        # Calculate differences
        diff = py_values - mat_values
        max_diff_abs = np.abs(diff).max()

        # Percentage difference (avoid division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            diff_pct = np.abs(diff / mat_values * 100)
            diff_pct = diff_pct[np.isfinite(diff_pct)]
            max_diff_pct = diff_pct.max() if len(diff_pct) > 0 else 0.0

        # Classify status
        if max_diff_pct < THRESHOLD_EXCELLENT:
            status = 'EXCELLENT'
        elif max_diff_pct < THRESHOLD_GOOD:
            status = 'GOOD'
        elif max_diff_pct < THRESHOLD_ACCEPTABLE:
            status = 'ACCEPTABLE'
        else:
            status = 'NEEDS REVIEW'

        return {
            'status': status,
            'max_diff_pct': max_diff_pct,
            'max_diff_abs': max_diff_abs,
            'worst_metric': 'array_values',
            'error_msg': None
        }

    except Exception as e:
        return {
            'status': 'ERROR',
            'max_diff_pct': np.nan,
            'max_diff_abs': np.nan,
            'worst_metric': 'N/A',
            'error_msg': str(e)
        }


def main():
    """Main comparison routine."""

    print("=" * 80)
    print("CSV Pair Comparison Report")
    print("=" * 80)
    print()

    # Find all CSV pairs
    pairs = find_csv_pairs(TEST_OUTPUT_DIR)

    if not pairs:
        print("[No CSV pairs found]")
        return

    # Apply filters
    if TEST_TYPES:
        pairs = [p for p in pairs if p['test_type'] in TEST_TYPES]
        print(f"[Filter] Test types: {TEST_TYPES}")

    if DATASETS:
        pairs = [p for p in pairs if p['dataset'] in DATASETS]
        print(f"[Filter] Datasets: {DATASETS}")

    if not pairs:
        print("[No pairs match filters]")
        return

    print(f"[Found] {len(pairs)} CSV pairs")
    print()

    # Compare all pairs
    results = []
    for pair in pairs:
        comparison = compare_csv_pair(pair['python_file'], pair['matlab_file'])
        results.append({
            **pair,
            **comparison
        })

    # Filter by problems only if requested
    if PROBLEMS_ONLY:
        results = [r for r in results if r['max_diff_pct'] >= THRESHOLD_GOOD]
        print(f"[Filter] Showing only problems (>={THRESHOLD_GOOD}% diff)")
        print()

    # Group by test type
    test_types = {}
    for r in results:
        test_type = r['test_type']
        if test_type not in test_types:
            test_types[test_type] = []
        test_types[test_type].append(r)

    # Count by status
    status_counts = {}
    for r in results:
        status = r['status']
        status_counts[status] = status_counts.get(status, 0) + 1

    # Print summary
    print("Summary by Test Type:")
    for test_type, items in sorted(test_types.items()):
        print(f"  {test_type:20s}: {len(items):3d} pairs")
    print()

    print("Summary by Status:")
    for status in ['EXCELLENT', 'GOOD', 'ACCEPTABLE', 'NEEDS REVIEW', 'ERROR']:
        count = status_counts.get(status, 0)
        pct = 100 * count / len(results) if len(results) > 0 else 0
        print(f"  {status:15s}: {count:3d} ({pct:5.1f}%)")
    print()

    # Print detailed results
    print("=" * 80)
    print("Detailed Results")
    print("=" * 80)
    print()

    for test_type, items in sorted(test_types.items()):
        print(f"[{test_type}]")
        print("-" * 80)

        for item in sorted(items, key=lambda x: x['dataset']):
            status_symbol = {
                'EXCELLENT': '✅',
                'GOOD': '✓',
                'ACCEPTABLE': '⚠️',
                'NEEDS REVIEW': '❌',
                'ERROR': '🔴'
            }.get(item['status'], '?')

            print(f"  {item['dataset']:50s} / {item['csv_name']:20s}")

            if item['error_msg']:
                print(f"    [ERROR] {item['error_msg']}")
            else:
                print(f"    Max diff: {item['max_diff_pct']:.3f}% (abs: {item['max_diff_abs']:.6e})")
                print(f"    Worst metric: {item['worst_metric']}")
                print(f"    Status: {item['status']} {status_symbol}")
            print()

        print()

    # Highlight files needing review
    needs_review = [r for r in results if r['status'] == 'NEEDS REVIEW']
    if needs_review:
        print("=" * 80)
        print("Files Needing Review")
        print("=" * 80)
        print()

        for idx, item in enumerate(needs_review, 1):
            print(f"{idx}. {item['pair_name']}")
            print(f"   Max diff: {item['max_diff_pct']:.2f}%")
            print(f"   Worst metric: {item['worst_metric']}")
            print()

    # Save summary to CSV
    summary_df = pd.DataFrame(results)
    summary_path = Path(TEST_OUTPUT_DIR) / 'comparison_summary_all.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"[Saved summary] -> {summary_path}")
    print()


if __name__ == "__main__":
    main()
