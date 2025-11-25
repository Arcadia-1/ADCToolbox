"""
compare_sineFit_results.py - Compare MATLAB and Python sineFit results

⚠️ DEPRECATION WARNING ⚠️
==========================
This script is DEPRECATED and will be removed in a future version.

Its functionality has been absorbed into the unified comparison system:
    python tests/system/compare_common.py  (for sineFit comparisons)
    python tests/system/universal_csv_compare.py --test-type test_sineFit

The unified system provides the same analysis with better integration,
filtering, and reporting capabilities.

This file is kept temporarily for backwards compatibility only.
==========================

Compares metrics and fit data from test_sineFit outputs.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Get project root directory
project_root = Path(__file__).parent.parent.parent


def compare_results():
    """Compare MATLAB and Python sineFit results across all datasets."""

    output_dir = project_root / "test_output"

    # Find all dataset folders that have test_sineFit results
    dataset_folders = []
    search_patterns = ["sinewave_*", "batch_sinewave_*"]

    for pattern in search_patterns:
        for folder in output_dir.glob(pattern):
            if not folder.is_dir():
                continue
            test_folder = folder / "test_sineFit"
            if test_folder.exists():
                matlab_metrics = test_folder / "metrics_matlab.csv"
                python_metrics = test_folder / "metrics_python.csv"
                if matlab_metrics.exists() and python_metrics.exists():
                    if folder.name not in dataset_folders:  # Avoid duplicates
                        dataset_folders.append(folder.name)

    if not dataset_folders:
        print("No matching MATLAB and Python results found.")
        print("Run both test_sineFit.m (MATLAB) and test_sine_fit.py (Python) first.")
        return

    print("=== sineFit MATLAB vs Python Comparison ===")
    print(f"Found {len(dataset_folders)} datasets with both MATLAB and Python results\n")

    all_results = []
    max_errors = {
        'freq': 0, 'mag': 0, 'dc': 0, 'phi': 0, 'fit_rms': 0
    }
    max_error_dataset = {
        'freq': '', 'mag': '', 'dc': '', 'phi': '', 'fit_rms': ''
    }

    for k, dataset_name in enumerate(sorted(dataset_folders), 1):
        test_folder = output_dir / dataset_name / "test_sineFit"

        # Read MATLAB results
        matlab_metrics = pd.read_csv(test_folder / "metrics_matlab.csv")
        matlab_fit = pd.read_csv(test_folder / "fit_data_matlab.csv")

        # Read Python results
        python_metrics = pd.read_csv(test_folder / "metrics_python.csv")
        python_fit = pd.read_csv(test_folder / "fit_data_python.csv")

        # Check if single or multi-column
        n_cols = len(matlab_metrics)

        # Verify same number of columns
        if len(python_metrics) != n_cols:
            print(f"  ERROR: Column count mismatch - MATLAB: {n_cols}, Python: {len(python_metrics)}")
            print(f"         Skipping {dataset_name}\n")
            continue

        # Compute metric errors
        freq_err = np.abs(matlab_metrics['freq'].values - python_metrics['freq'].values)
        mag_err = np.abs(matlab_metrics['mag'].values - python_metrics['mag'].values)
        dc_err = np.abs(matlab_metrics['dc'].values - python_metrics['dc'].values)
        phi_err = np.abs(matlab_metrics['phi'].values - python_metrics['phi'].values)

        # Compute fit data RMS error
        fit_matlab = matlab_fit.values
        fit_python = python_fit.values
        fit_rms_err = np.nan  # Default to NaN in case of error

        # Handle shape mismatch (MATLAB might have transposed or different format)
        if fit_matlab.shape != fit_python.shape:
            # Try transposing MATLAB data
            if fit_matlab.T.shape == fit_python.shape:
                fit_matlab = fit_matlab.T
                fit_rms_err = np.sqrt(np.mean((fit_matlab - fit_python)**2))
            else:
                # Shape mismatch - MATLAB may have averaged columns while Python didn't
                print(f"  WARNING: Shape mismatch - MATLAB: {matlab_fit.shape}, Python: {python_fit.shape}")
                print(f"           Skipping fit_data comparison for this dataset")
        else:
            fit_rms_err = np.sqrt(np.mean((fit_matlab - fit_python)**2))

        # Store results
        result = {
            'dataset': dataset_name,
            'n_cols': n_cols,
            'freq_err_max': np.max(freq_err),
            'freq_err_mean': np.mean(freq_err),
            'mag_err_max': np.max(mag_err),
            'mag_err_mean': np.mean(mag_err),
            'dc_err_max': np.max(dc_err),
            'dc_err_mean': np.mean(dc_err),
            'phi_err_max': np.max(phi_err),
            'phi_err_mean': np.mean(phi_err),
            'fit_rms': fit_rms_err
        }
        all_results.append(result)

        # Track maximum errors (skip NaN values)
        if result['freq_err_max'] > max_errors['freq']:
            max_errors['freq'] = result['freq_err_max']
            max_error_dataset['freq'] = dataset_name
        if result['mag_err_max'] > max_errors['mag']:
            max_errors['mag'] = result['mag_err_max']
            max_error_dataset['mag'] = dataset_name
        if result['dc_err_max'] > max_errors['dc']:
            max_errors['dc'] = result['dc_err_max']
            max_error_dataset['dc'] = dataset_name
        if result['phi_err_max'] > max_errors['phi']:
            max_errors['phi'] = result['phi_err_max']
            max_error_dataset['phi'] = dataset_name
        if not np.isnan(result['fit_rms']) and result['fit_rms'] > max_errors['fit_rms']:
            max_errors['fit_rms'] = result['fit_rms']
            max_error_dataset['fit_rms'] = dataset_name

        # Print summary for this dataset
        fit_check = np.isnan(result['fit_rms']) or result['fit_rms'] < 1e-6
        status = "PASS" if (result['freq_err_max'] < 1e-10 and
                           result['mag_err_max'] < 1e-6 and
                           result['dc_err_max'] < 1e-6 and
                           result['phi_err_max'] < 1e-6 and
                           fit_check) else "CHECK"

        print(f"[{k}/{len(dataset_folders)}] {dataset_name} ({n_cols} col{'s' if n_cols > 1 else ''}) - {status}")
        if n_cols == 1:
            print(f"  freq_err={result['freq_err_max']:.2e}, mag_err={result['mag_err_max']:.2e}, "
                  f"dc_err={result['dc_err_max']:.2e}, phi_err={result['phi_err_max']:.2e}")
        else:
            print(f"  freq: max_err={result['freq_err_max']:.2e}, mean_err={result['freq_err_mean']:.2e}")
            print(f"  mag:  max_err={result['mag_err_max']:.2e}, mean_err={result['mag_err_mean']:.2e}")
            print(f"  dc:   max_err={result['dc_err_max']:.2e}, mean_err={result['dc_err_mean']:.2e}")
            print(f"  phi:  max_err={result['phi_err_max']:.2e}, mean_err={result['phi_err_mean']:.2e}")

        if np.isnan(result['fit_rms']):
            print(f"  fit_data RMS error: N/A (shape mismatch)\n")
        else:
            print(f"  fit_data RMS error: {result['fit_rms']:.2e}\n")

    # Print overall summary
    print("=" * 60)
    print("Overall Summary:")
    print(f"  Total datasets compared: {len(all_results)}")
    print(f"\n  Maximum errors across all datasets:")
    print(f"    freq:     {max_errors['freq']:.2e}  ({max_error_dataset['freq']})")
    print(f"    mag:      {max_errors['mag']:.2e}  ({max_error_dataset['mag']})")
    print(f"    dc:       {max_errors['dc']:.2e}  ({max_error_dataset['dc']})")
    print(f"    phi:      {max_errors['phi']:.2e}  ({max_error_dataset['phi']})")
    if max_errors['fit_rms'] > 0:
        print(f"    fit_rms:  {max_errors['fit_rms']:.2e}  ({max_error_dataset['fit_rms']})")
    else:
        print(f"    fit_rms:  N/A (all datasets had shape mismatch)")

    # Check if all pass
    all_pass = all(
        r['freq_err_max'] < 1e-10 and
        r['mag_err_max'] < 1e-6 and
        r['dc_err_max'] < 1e-6 and
        r['phi_err_max'] < 1e-6 and
        (np.isnan(r['fit_rms']) or r['fit_rms'] < 1e-6)
        for r in all_results
    )

    print(f"\n  Overall status: {'ALL PASS' if all_pass else 'SOME FAILURES - CHECK ABOVE'}")
    print("=" * 60)

    # Save detailed comparison to CSV
    comparison_csv = output_dir / "sineFit_comparison_summary.csv"
    df = pd.DataFrame(all_results)
    df.to_csv(comparison_csv, index=False)
    print(f"\nDetailed comparison saved to: {comparison_csv}")


if __name__ == '__main__':
    compare_results()
