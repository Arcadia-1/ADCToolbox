"""
Run all unit tests for common package

Configuration - assumes running from project root d:\ADCToolbox

This script runs unit tests for common ADC analysis functions:
- alias (frequency aliasing)
- sineFit (sine wave fitting)
"""

import subprocess
import sys
from pathlib import Path


def run_unit_tests_common():
    """Run all common package unit tests."""
    print("=" * 80)
    print("Running Unit Tests - Common Package")
    print("=" * 80)
    print()

    # List of unit tests to run
    unit_tests = [
        "test_alias.py",
        "test_sine_fit.py",
    ]

    # Base directory for unit tests
    unit_test_dir = Path("tests/unit")

    results = []
    for test_file in unit_tests:
        test_path = unit_test_dir / test_file
        print(f"Running: {test_file}")
        print("-" * 80)

        result = subprocess.run(
            [sys.executable, str(test_path)],
            capture_output=True,
            text=True
        )

        success = (result.returncode == 0)
        results.append((test_file, success))

        if success:
            print(f"[FINISHED] {test_file}")
        else:
            print(f"[FAIL] {test_file}")
            print(result.stderr)

        print()

    # Summary
    print("=" * 80)
    print("Summary - Common Package Unit Tests")
    print("=" * 80)
    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_file, success in results:
        status = "FINISHED" if success else "FAIL"
        print(f"  [{status}] {test_file}")

    print()
    print(f"[Total] {passed}/{total} tests passed")
    print("=" * 80)

    return passed == total


if __name__ == "__main__":
    success = run_unit_tests_common()
    sys.exit(0 if success else 1)
