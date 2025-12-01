"""
Run all unit tests for entire ADCToolbox package

Configuration - assumes running from project root d:\ADCToolbox

This is the top-level system test that runs all package-specific unit tests:
- Common package tests (alias, sineFit)
- Aout package tests (analog output analysis)
- Dout package tests (digital output analysis)

Usage:
    cd d:\ADCToolbox
    python tests/system/run_unit_tests_all.py
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime


def run_unit_tests_all():
    """Run all unit tests across all packages."""
    print("=" * 80)
    print("ADCToolbox - Complete Unit Test Suite")
    print("=" * 80)
    print(f"[Started] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # System test directory
    system_test_dir = Path("tests/system")

    # List of package test runners (in order)
    package_tests = [
        ("Common Package", "run_unit_tests_common.py"),
        ("Aout Package", "run_unit_tests_aout.py"),
        ("Dout Package", "run_unit_tests_dout.py"),
    ]

    results = []
    start_time = time.time()

    for package_name, test_file in package_tests:
        test_path = system_test_dir / test_file

        print()
        print("=" * 80)
        print(f"Package: {package_name}")
        print("=" * 80)
        print()

        pkg_start = time.time()

        result = subprocess.run(
            [sys.executable, str(test_path)],
            capture_output=False,  # Show output in real-time
            text=True
        )

        pkg_time = time.time() - pkg_start
        success = (result.returncode == 0)
        results.append((package_name, success, pkg_time))

        print()
        if success:
            print(f"[FINISHED] {package_name} ({pkg_time:.1f}s)")
        else:
            print(f"[FAIL] {package_name} ({pkg_time:.1f}s)")
        print()

    total_time = time.time() - start_time

    # Final Summary
    print()
    print("=" * 80)
    print("FINAL SUMMARY - All Unit Tests")
    print("=" * 80)
    print()

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    for package_name, success, pkg_time in results:
        status = "FINISHED" if success else "FAIL"
        print(f"  [{status}] {package_name:30s} ({pkg_time:6.1f}s)")

    print()
    print(f"[Packages passed]      = {passed}/{total}")
    print(f"[Success rate]         = {passed/total*100:.1f}%")
    print(f"[Total execution time] = {total_time:.1f} seconds")
    print()

    if passed == total:
        print("=" * 80)
        print("ALL UNIT TESTS PASSED!")
        print("=" * 80)
    else:
        print("=" * 80)
        print(f"WARNING: {total - passed} package(s) failed")
        print("=" * 80)

    return passed == total


if __name__ == "__main__":
    success = run_unit_tests_all()
    sys.exit(0 if success else 1)
