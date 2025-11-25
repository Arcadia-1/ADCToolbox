"""
Run all unit tests for aout package (analog output analysis)

Configuration - assumes running from project root d:\ADCToolbox

This script runs unit tests for analog output analysis functions:
- tomDecomp (time-domain decomposition)
- specPlot (spectrum analysis)
- specPlotPhase (phase spectrum)
- errHistSine (error histogram)
- errPDF (error probability density)
- errAutoCorrelation (error autocorrelation)
- errEnvelopeSpectrum (envelope spectrum)
- INLSine (INL from sine wave)
"""

import subprocess
import sys
from pathlib import Path


def run_unit_tests_aout():
    """Run all aout package unit tests."""
    print("=" * 80)
    print("Running Unit Tests - Aout Package (Analog Output Analysis)")
    print("=" * 80)
    print()

    # List of unit tests to run (in order)
    unit_tests = [
        "test_specPlot.py",
        "test_specPlotPhase.py",
        "test_error_analysis.py",  # Combines errPDF, errAutoCorrelation, errEnvelopeSpectrum
        "test_INLSine.py",
    ]

    # Base directory for unit tests
    unit_test_dir = Path("adctoolbox/test/unit")

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
            print(f"[PASS] {test_file}")
        else:
            print(f"[FAIL] {test_file}")
            print(result.stderr)

        print()

    # Summary
    print("=" * 80)
    print("Summary - Aout Package Unit Tests")
    print("=" * 80)
    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_file, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {test_file}")

    print()
    print(f"[Total] {passed}/{total} tests passed")
    print("=" * 80)

    return passed == total


if __name__ == "__main__":
    success = run_unit_tests_aout()
    sys.exit(0 if success else 1)
