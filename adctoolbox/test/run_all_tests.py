"""
ADCToolbox Test Runner - Run complete test suite with comprehensive report

This is the top-level test runner that executes the system test suite
and generates a detailed report.

Test Hierarchy:
    run_all_tests.py (this file)
        └── system/run_unit_tests_all.py
            ├── system/run_unit_tests_common.py
            │   ├── unit/test_alias.py
            │   └── unit/test_sineFit.py
            ├── system/run_unit_tests_aout.py
            │   ├── unit/test_specPlot.py
            │   ├── unit/test_specPlotPhase.py
            │   ├── unit/test_error_analysis.py
            │   └── unit/test_INLSine.py
            └── system/run_unit_tests_dout.py
                ├── unit/test_FGCalSine.py
                ├── unit/test_FGCalSine_overflowChk.py
                └── unit/test_cap2weight.py

Configuration - assumes running from project root d:\ADCToolbox

Usage:
    cd d:\ADCToolbox
    python adctoolbox/test/run_all_tests.py
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
import re


# ============================================================================
# Configuration
# ============================================================================

# Output report file
REPORT_FILE = "test_output/TEST_REPORT.txt"


# ============================================================================
# Test Execution
# ============================================================================

def run_system_tests():
    """Run the complete system test suite."""
    print("=" * 80)
    print("ADCToolbox Python Test Suite Runner")
    print("=" * 80)
    print(f"[Started] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Run the top-level system test
    system_test = Path("adctoolbox/test/system/run_unit_tests_all.py")

    start_time = time.time()

    result = subprocess.run(
        [sys.executable, str(system_test)],
        capture_output=True,
        text=True
    )

    total_time = time.time() - start_time

    return result, total_time


def parse_test_output(output):
    """Parse test output to extract metrics."""
    metrics = {
        'packages_passed': 0,
        'packages_total': 0,
        'success_rate': 0.0,
        'package_results': []
    }

    # Extract package results
    for line in output.split('\n'):
        # Match lines like: "  [PASS] Common Package                 (  2.5s)"
        match = re.match(r'\s*\[(PASS|FAIL)\]\s+(.+?)\s+\((.+?)s\)', line)
        if match:
            status, package_name, pkg_time = match.groups()
            success = (status == 'PASS')
            metrics['package_results'].append({
                'name': package_name.strip(),
                'success': success,
                'time': float(pkg_time)
            })

        # Match total passed line
        match = re.search(r'\[Packages passed\]\s*=\s*(\d+)/(\d+)', line)
        if match:
            metrics['packages_passed'] = int(match.group(1))
            metrics['packages_total'] = int(match.group(2))

        # Match success rate
        match = re.search(r'\[Success rate\]\s*=\s*([\d.]+)%', line)
        if match:
            metrics['success_rate'] = float(match.group(1))

    return metrics


def generate_report(result, total_time, metrics, report_path):
    """Generate comprehensive test report."""
    report_lines = []

    # Header
    report_lines.append("=" * 80)
    report_lines.append("ADCToolbox Python Test Suite - Comprehensive Report")
    report_lines.append("=" * 80)
    report_lines.append(f"[Generated] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    # Test Hierarchy
    report_lines.append("=" * 80)
    report_lines.append("Test Hierarchy")
    report_lines.append("=" * 80)
    report_lines.append("run_all_tests.py (top level)")
    report_lines.append("  └── system/run_unit_tests_all.py")
    report_lines.append("      ├── system/run_unit_tests_common.py")
    report_lines.append("      │   ├── unit/test_alias.py")
    report_lines.append("      │   └── unit/test_sineFit.py")
    report_lines.append("      ├── system/run_unit_tests_aout.py")
    report_lines.append("      │   ├── unit/test_specPlot.py")
    report_lines.append("      │   ├── unit/test_specPlotPhase.py")
    report_lines.append("      │   ├── unit/test_error_analysis.py")
    report_lines.append("      │   └── unit/test_INLSine.py")
    report_lines.append("      └── system/run_unit_tests_dout.py")
    report_lines.append("          ├── unit/test_FGCalSine.py")
    report_lines.append("          ├── unit/test_FGCalSine_overflowChk.py")
    report_lines.append("          └── unit/test_cap2weight.py")
    report_lines.append("")

    # Summary Statistics
    report_lines.append("=" * 80)
    report_lines.append("Summary Statistics")
    report_lines.append("=" * 80)
    report_lines.append(f"[Packages tested]      = {metrics['packages_total']}")
    report_lines.append(f"[Packages passed]      = {metrics['packages_passed']}")
    report_lines.append(f"[Packages failed]      = {metrics['packages_total'] - metrics['packages_passed']}")
    report_lines.append(f"[Success rate]         = {metrics['success_rate']:.1f}%")
    report_lines.append(f"[Total execution time] = {total_time:.1f} seconds")
    report_lines.append("")

    # Package Results
    if metrics['package_results']:
        report_lines.append("=" * 80)
        report_lines.append("Package Test Results")
        report_lines.append("=" * 80)

        for pkg in metrics['package_results']:
            status = "PASS" if pkg['success'] else "FAIL"
            report_lines.append(f"  [{status}] {pkg['name']:30s} ({pkg['time']:6.1f}s)")

        report_lines.append("")

    # Full Output
    report_lines.append("=" * 80)
    report_lines.append("Complete Test Output")
    report_lines.append("=" * 80)
    report_lines.append(result.stdout)
    report_lines.append("")

    # Errors (if any)
    if result.stderr:
        report_lines.append("=" * 80)
        report_lines.append("Error Output")
        report_lines.append("=" * 80)
        report_lines.append(result.stderr)
        report_lines.append("")

    # Footer
    report_lines.append("=" * 80)
    if result.returncode == 0:
        report_lines.append("ALL TESTS PASSED!")
    else:
        report_lines.append(f"WARNING: Some tests failed (exit code: {result.returncode})")
    report_lines.append("=" * 80)

    # Write report to file
    report_text = '\n'.join(report_lines)

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    return report_text


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    """Main test runner."""
    # Run system tests
    result, total_time = run_system_tests()

    # Parse output for metrics
    metrics = parse_test_output(result.stdout)

    # Generate report
    print()
    print("=" * 80)
    print("Generating comprehensive report...")
    print("=" * 80)

    generate_report(result, total_time, metrics, REPORT_FILE)

    # Print summary to console
    print()
    print("=" * 80)
    print("Test Run Complete")
    print("=" * 80)
    print(f"[Packages tested]      = {metrics['packages_total']}")
    print(f"[Packages passed]      = {metrics['packages_passed']}")
    print(f"[Success rate]         = {metrics['success_rate']:.1f}%")
    print(f"[Total execution time] = {total_time:.1f} seconds")
    print()
    print(f"[Report saved to]      = {REPORT_FILE}")
    print("=" * 80)

    # Return success/failure
    return result.returncode == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
