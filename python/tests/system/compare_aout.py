"""
Compare MATLAB vs Python CSV outputs for Aout Package tests

Configuration - assumes running from project root d:\ADCToolbox

Tests in Aout Package (Analog Output Analysis):
- test_specPlot
- test_specPlotPhase
- test_errPDF
- test_errAutoCorrelation
- test_errEnvelopeSpectrum
- test_errSpectrum
- test_errHistSine
- test_tomDecomp
- test_INLSine

Usage:
    cd d:\ADCToolbox
    python tests/system/compare_aout.py
"""

import sys
from pathlib import Path

# Add project root to path for importing tests.utils
if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.utils import CSVComparator


def compare_aout():
    """Compare CSV outputs for aout package tests."""
    print("=" * 80)
    print("CSV Comparison - Aout Package (Analog Output Analysis)")
    print("=" * 80)
    print()

    # Test types in aout package
    test_types = [
        'test_specPlot',
        'test_specPlotPhase',
        'test_errPDF',
        'test_errAutoCorrelation',
        'test_errEnvelopeSpectrum',
        'test_errSpectrum',
        'test_errHistSine',
        'test_tomDecomp',
        'test_INLSine',
    ]

    print(f"[Test types] {', '.join(test_types)}")
    print()

    # Create comparator
    comparator = CSVComparator(test_output_dir='test_output')

    # Find all pairs
    all_pairs = comparator.find_csv_pairs()

    # Filter to aout package test types
    aout_pairs = [p for p in all_pairs if p['test_type'] in test_types]

    if not aout_pairs:
        print("[No CSV pairs found for aout package]")
        return True

    print(f"[Found] {len(aout_pairs)} CSV pairs in aout package")
    print()

    # Run comparisons with progress display
    results = []
    total = len(aout_pairs)
    for i, pair in enumerate(aout_pairs, 1):
        comparison = comparator.compare_pair(pair['python_file'], pair['matlab_file'])
        results.append({
            **pair,
            **comparison
        })

        # Print formatted progress with fixed-width status
        status = comparison['status']
        status_padded = f"{status:15s}"  # Fixed width of 15 chars
        max_diff_pct = comparison['max_diff_pct']
        csv_name = f"{pair['pair_name']}.csv"

        print(f"[{i:3d}/{total:3d}][{status_padded}] (diff: {max_diff_pct:8.4f}%) {csv_name}")

    print()

    # Create results DataFrame
    import pandas as pd
    results_df = pd.DataFrame(results)

    # Save results
    output_path = Path('test_output/comparison_aout.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    # Print summary
    print("=" * 80)
    print("Summary - Aout Package")
    print("=" * 80)
    print()

    # Grading criteria and results (combined)
    print("[Results by Grade]")
    for status, threshold in [('PERFECT', '< 1e-10%'), ('EXCELLENT', '< 0.01%'),
                              ('GOOD', '< 0.1%'), ('ACCEPTABLE', '< 1.0%'),
                              ('NEEDS REVIEW', '≥ 1.0%'), ('ERROR', 'N/A')]:
        count = (results_df['status'] == status).sum()
        if count > 0:
            pct = 100 * count / len(results_df)
            print(f"  [{threshold:>8s}] {status:12s} : {count:3d} ({pct:5.1f}%)")
    print()

    # Files needing review
    needs_review = results_df[results_df['status'] == 'NEEDS REVIEW']
    if not needs_review.empty:
        # Separate into real issues vs near-zero artifacts
        real_issues = needs_review[needs_review['note'].isna()]
        near_zero_artifacts = needs_review[needs_review['note'].notna()]

        if not real_issues.empty:
            print(f"[WARNING] {len(real_issues)} file(s) need review (real issues):")
            for _, row in real_issues.iterrows():
                print(f"  - {row['pair_name']} (diff: {row['max_diff_pct']:.4f}%, abs: {row['max_diff_abs']:.2e})")
            print()

        if not near_zero_artifacts.empty:
            print(f"[INFO] {len(near_zero_artifacts)} file(s) flagged but are near-zero artifacts (OK):")
            print(f"  (Large percentages on tiny absolute differences < 1e-5)")
            print()

    # Errors
    errors = results_df[results_df['status'] == 'ERROR']
    if not errors.empty:
        print(f"[ERROR] {len(errors)} file(s) had errors:")
        for _, row in errors.iterrows():
            print(f"  - {row['pair_name']}: {row['error_msg']}")
        print()

    print(f"[Results saved] {output_path}")
    print("=" * 80)
    print()

    # Return success if no files need review or have errors
    return needs_review.empty and errors.empty


if __name__ == "__main__":
    success = compare_aout()
    sys.exit(0 if success else 1)
