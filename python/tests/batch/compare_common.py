"""
Compare MATLAB vs Python CSV outputs for Common Package tests

Configuration - assumes running from project root d:\ADCToolbox

Tests in Common Package:
- test_alias
- test_sineFit

Usage:
    cd d:\ADCToolbox
    python tests/system/compare_common.py
"""

import sys
from pathlib import Path

# Add project root to path for importing tests.utils
if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.utils import CSVComparator


def compare_common():
    """Compare CSV outputs for common package tests."""
    print("=" * 80)
    print("CSV Comparison - Common Package")
    print("=" * 80)
    print()

    # Test types in common package
    test_types = [
        # 'test_alias',
        'test_sineFit',
    ]

    print(f"[Test types] {', '.join(test_types)}")
    print()

    # Create comparator (use absolute path from this file's location)
    script_dir = Path(__file__).resolve().parent  # python/tests/system
    project_root = script_dir.parent.parent.parent  # d:\ADCToolbox
    test_output_dir = project_root / 'test_output'
    comparator = CSVComparator(test_output_dir=str(test_output_dir))

    # Find all pairs
    all_pairs = comparator.find_csv_pairs()

    # Filter to common package test types
    common_pairs = [p for p in all_pairs if p['test_type'] in test_types]

    if not common_pairs:
        print("[WARNING] No CSV pairs found for common package")
        print("[INFO] MATLAB reference files (*_matlab.csv) not found")
        print("[INFO] Run MATLAB tests first to generate reference outputs")
        return None  # Return None to indicate no data to compare

    print(f"[Found] {len(common_pairs)} CSV pairs in common package")
    print()

    # Run comparisons with progress display
    results = []
    total = len(common_pairs)
    for i, pair in enumerate(common_pairs, 1):
        comparison = comparator.compare_pair(pair['python_file'], pair['matlab_file'])
        results.append({
            **pair,
            **comparison
        })

        # Print formatted progress with fixed-width status
        status = comparison['status']
        status_padded = f"{status:12s}"  # Fixed width of 15 chars
        max_diff_pct = comparison['max_diff_pct']
        csv_name = f"{pair['pair_name']}.csv"

        print(f"[{i:3d}/{total:3d}][{status_padded}] (diff: {max_diff_pct:8.4f}%) {csv_name}")

    print()

    # Create results DataFrame
    import pandas as pd
    results_df = pd.DataFrame(results)

    # Save results
    output_path = test_output_dir / 'comparison_common.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    # Print summary
    print("=" * 80)
    print("Summary - Common Package")
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
        print(f"[WARNING] {len(needs_review)} file(s) need review:")
        for _, row in needs_review.iterrows():
            print(f"  - {row['pair_name']} (diff: {row['max_diff_pct']:.4f}%)")
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
    success = compare_common()
    sys.exit(0 if success else 1)
