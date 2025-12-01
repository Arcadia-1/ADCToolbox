"""universal_csv_compare.py - Universal CSV comparison tool

Compares MATLAB vs Python CSV outputs for any test type.
Supports both metrics-style and array-style CSV files.

Usage:
    python tests/system/universal_csv_compare.py

Features:
    - Auto-discovers all *_matlab.csv and *_python.csv pairs
    - Handles metrics files (with 'Metric' column)
    - Handles array/time-series files (numeric data)
    - Generates comprehensive comparison reports
    - Identifies files needing review
"""

import sys
from pathlib import Path

# Add project root to path for importing tests.utils
if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.utils import CSVComparator


def main():
    """Main comparison routine."""
    import argparse

    parser = argparse.ArgumentParser(description='Compare MATLAB vs Python CSV outputs')
    parser.add_argument('--test-type', nargs='+', help='Filter by test type(s)')
    parser.add_argument('--dataset', nargs='+', help='Filter by dataset(s)')
    parser.add_argument('--output', default='test_output/comparison_summary.csv',
                       help='Output CSV file path')

    args = parser.parse_args()

    # Create comparator
    comparator = CSVComparator()

    # Run comparisons
    results_df = comparator.compare_all(test_types=args.test_type, datasets=args.dataset)

    if results_df.empty:
        return 1

    # Print summary
    comparator.print_summary(results_df)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"[Saved] {output_path}")
    print()

    # Exit code based on results
    needs_review = (results_df['status'] == 'NEEDS REVIEW').sum()
    errors = (results_df['status'] == 'ERROR').sum()

    if errors > 0:
        return 2
    elif needs_review > 0:
        return 1
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
