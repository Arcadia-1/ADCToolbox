"""
Compare MATLAB vs Python CSV outputs for test_basic

Configuration - assumes running from project root d:\ADCToolbox

Tests:
- test_basic - Basic sine wave generation

Usage:
    cd d:\ADCToolbox
    python python/tests/system/compare_basic.py
"""

import sys
from pathlib import Path

# Add project root to path for importing tests.utils
if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Get project root directory (three levels up from python/tests/unit)
project_root = Path(__file__).resolve().parents[3]
print(f"[INFO] Project root directory: {project_root}")


# from tests.utils import CSVComparator


def compare_basic():
    """Compare CSV outputs for test_basic."""
    print("=" * 80)
    print("CSV Comparison - test_basic")
    print("=" * 80)
    print()

    # Dataset names to compare
    datasets = ['test_basic']

    print(f"[Datasets] {', '.join(datasets)}")
    print()

    # Create comparator (use absolute path from this file's location)
    script_dir = Path(__file__).resolve().parent  # python/tests/system
    project_root = script_dir.parent.parent.parent  # d:\ADCToolbox
    test_output_dir = project_root / 'test_output'
    comparator = CSVComparator(test_output_dir=str(test_output_dir))

    # Find all pairs
    all_pairs = comparator.find_csv_pairs()

    # Filter to test_basic dataset
    filtered_pairs = [p for p in all_pairs if p['dataset'] in datasets]

    if not filtered_pairs:
        print(f"[WARNING] No CSV pairs found for {datasets}")
        print(f"[INFO] Search directory: {test_output_dir}")
        print(f"[INFO] Run the following first:")
        print(f"  MATLAB: matlab/tests/unit/test_basic.m")
        print(f"  Python: python/tests/unit/test_basic.py")
        return

    print(f"[Found] {len(filtered_pairs)} CSV pair(s)")
    for pair in filtered_pairs:
        print(f"  - {pair['pair_name']}")
    print()

    # Compare all pairs
    print("=" * 80)
    print("Comparing CSV Files")
    print("=" * 80)
    print()

    results = []
    total = len(filtered_pairs)
    for i, pair in enumerate(filtered_pairs, 1):
        comparison = comparator.compare_pair(pair['python_file'], pair['matlab_file'])

        # Check against stricter tolerance for test_basic (no noise, expect exact match)
        max_diff = comparison.get('max_diff', float('inf'))
        if max_diff < 1e-14:
            comparison['status'] = 'PASS'
        elif comparison['status'] == 'PASS':  # Was passing with default tolerance but not strict
            comparison['status'] = 'FAIL'
            comparison['message'] = f"Max diff {max_diff:.2e} exceeds strict tolerance 1e-14"

        results.append({
            **pair,
            **comparison
        })

        # Print formatted progress
        status = comparison['status']
        status_padded = f"{status:12s}"
        print(f"  [{i}/{total}] {status_padded} {pair['pair_name']}")
        if status == 'FAIL':
            print(f"       Max diff: {comparison.get('max_diff', 'N/A')}")
        elif status == 'ERROR':
            print(f"       Error: {comparison.get('message', 'Unknown error')}")

    # Convert to dict for summary
    results_dict = {r['pair_name']: r for r in results}

    # Summary
    print()
    print("=" * 80)
    print("Comparison Summary")
    print("=" * 80)
    total = len(results)
    passed = sum(1 for r in results if r['status'] == 'PASS')
    failed = sum(1 for r in results if r['status'] == 'FAIL')
    errors = sum(1 for r in results if r['status'] == 'ERROR')
    skipped = sum(1 for r in results if r['status'] == 'SKIP')

    print(f"Total: {total}, PASS: {passed}, FAIL: {failed}, ERROR: {errors}, SKIP: {skipped}")

    if failed > 0 or errors > 0:
        print(f"\n[FAIL] {failed} failed, {errors} errors!")
        sys.exit(1)
    elif skipped == total:
        print("\n[SKIP] All comparisons skipped (files not found)")
        sys.exit(0)
    else:
        print("\n[PASS] All comparisons passed!")
        sys.exit(0)


if __name__ == "__main__":
    compare_basic()
