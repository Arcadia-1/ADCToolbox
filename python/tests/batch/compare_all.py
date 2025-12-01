"""
Compare all MATLAB vs Python CSV outputs (all packages)

Configuration - assumes running from project root d:\ADCToolbox

This top-level comparison script runs CSV comparison for all packages:
- Common package (alias, sineFit)
- Aout package (analog output analysis)
- Dout package (digital output analysis)

Usage:
    cd d:\ADCToolbox
    python tests/system/compare_all.py
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path for importing tests modules
if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import package-specific comparators
from tests.system.compare_common import compare_common
from tests.system.compare_aout import compare_aout
from tests.system.compare_dout import compare_dout


def compare_all():
    """Run CSV comparison for all packages."""
    print("=" * 80)
    print("ADCToolbox - Complete CSV Comparison Suite")
    print("=" * 80)
    print(f"[Started] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Package comparison functions
    package_comparisons = [
        ("Common Package", compare_common),
        ("Aout Package", compare_aout),
        ("Dout Package", compare_dout),
    ]

    results = []
    start_time = time.time()

    for package_name, compare_func in package_comparisons:
        print()
        print("=" * 80)
        print(f"Package: {package_name}")
        print("=" * 80)
        print()

        pkg_start = time.time()

        try:
            success = compare_func()
            pkg_time = time.time() - pkg_start
            results.append((package_name, success, pkg_time, None))

            print()
            if success is None:
                print(f"[SKIPPED] {package_name} - No MATLAB reference files ({pkg_time:.1f}s)")
            elif success:
                print(f"[PASS] {package_name} ({pkg_time:.1f}s)")
            else:
                print(f"[FAIL] {package_name} - Some files need review ({pkg_time:.1f}s)")
        except Exception as e:
            pkg_time = time.time() - pkg_start
            results.append((package_name, False, pkg_time, str(e)))
            print(f"[ERROR] {package_name} - {str(e)} ({pkg_time:.1f}s)")

        print()

    total_time = time.time() - start_time

    # Final Summary
    print()
    print("=" * 80)
    print("FINAL SUMMARY - All CSV Comparisons")
    print("=" * 80)
    print()

    passed = sum(1 for _, success, _, _ in results if success is True)
    skipped = sum(1 for _, success, _, _ in results if success is None)
    failed = sum(1 for _, success, _, _ in results if success is False)
    total = len(results)

    for package_name, success, pkg_time, error in results:
        if success is None:
            status = "SKIPPED"
        elif success:
            status = "PASS"
        else:
            status = "FAIL"
        print(f"  [{status}] {package_name:30s} ({pkg_time:6.1f}s)")
        if error:
            print(f"        Error: {error}")

    print()
    print(f"[Packages passed]      = {passed}/{total}")
    print(f"[Packages skipped]     = {skipped}/{total}")
    print(f"[Packages failed]      = {failed}/{total}")
    if passed + failed > 0:
        print(f"[Success rate]         = {passed/(passed+failed)*100:.1f}% (of tested)")
    else:
        print(f"[Success rate]         = N/A (no comparisons performed)")
    print(f"[Total execution time] = {total_time:.1f} seconds")
    print()

    # List output files
    print("=" * 80)
    print("Output Files")
    print("=" * 80)
    print()
    output_files = [
        'test_output/comparison_common.csv',
        'test_output/comparison_aout.csv',
        'test_output/comparison_dout.csv',
    ]
    for output_file in output_files:
        if Path(output_file).exists():
            print(f"  - {output_file}")
    print()

    print("=" * 80)
    if skipped == total:
        print("NO COMPARISONS PERFORMED - MATLAB reference files missing")
        print("Run MATLAB tests first: cd matlab && run_unit_tests_all")
    elif failed == 0 and passed > 0:
        print("ALL COMPARISONS PASSED!")
    elif failed > 0:
        print(f"WARNING: {failed} package(s) failed comparison")
    print("=" * 80)

    return passed > 0 and failed == 0


def main():
    """Main entry point with comprehensive report generation."""
    print()

    # Run all comparisons
    success = compare_all()

    # Generate master summary report
    print()
    print("=" * 80)
    print("Generating master comparison report...")
    print("=" * 80)
    print()

    try:
        import pandas as pd

        # Combine all package results
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ADCToolbox - Master CSV Comparison Report")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Load each package's results
        all_results = []
        for csv_file in ['comparison_common.csv', 'comparison_aout.csv', 'comparison_dout.csv']:
            csv_path = Path('test_output') / csv_file
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                package_name = csv_file.replace('comparison_', '').replace('.csv', '')
                df['package'] = package_name
                all_results.append(df)

        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)

            # Overall statistics
            report_lines.append("=" * 80)
            report_lines.append("Overall Statistics")
            report_lines.append("=" * 80)
            report_lines.append(f"Total CSV pairs compared: {len(combined_df)}")
            report_lines.append("")

            # By package
            report_lines.append("By Package:")
            for package, group in combined_df.groupby('package'):
                report_lines.append(f"  {package:10s}: {len(group):3d} pairs")
            report_lines.append("")

            # By status
            report_lines.append("By Status:")
            for status in ['PERFECT', 'EXCELLENT', 'GOOD', 'ACCEPTABLE', 'NEEDS REVIEW', 'ERROR']:
                count = (combined_df['status'] == status).sum()
                if count > 0:
                    pct = 100 * count / len(combined_df)
                    report_lines.append(f"  {status:15s}: {count:3d} ({pct:5.1f}%)")
            report_lines.append("")

            # Files needing review
            needs_review = combined_df[combined_df['status'] == 'NEEDS REVIEW']
            if not needs_review.empty:
                report_lines.append("=" * 80)
                report_lines.append(f"Files Needing Review ({len(needs_review)})")
                report_lines.append("=" * 80)
                report_lines.append("")
                for _, row in needs_review.iterrows():
                    report_lines.append(f"[FAIL] {row['pair_name']}")
                    report_lines.append(f"   Package: {row['package']}")
                    report_lines.append(f"   Max diff: {row['max_diff_pct']:.4f}%")
                    report_lines.append(f"   Worst field: {row['worst_field']}")
                    report_lines.append("")

            # Save report
            report_path = Path('test_output/COMPARISON_REPORT.txt')
            with open(report_path, 'w') as f:
                f.write('\n'.join(report_lines))

            print(f"[Master report saved] {report_path}")
            print()

    except Exception as e:
        print(f"[Warning] Could not generate master report: {e}")
        print()

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
