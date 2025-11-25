"""
Run all unit tests for aout package (analog output analysis)

Configuration - assumes running from project root d:\ADCToolbox

This script runs unit tests for analog output analysis functions:
- specPlot (spectrum analysis)
- specPlotPhase (phase spectrum)
- errPDF (error probability density)
- errAutoCorrelation (error autocorrelation)
- errEnvelopeSpectrum (envelope spectrum)
- errSpectrum (error spectrum)
- errHistSine (error histogram with jitter detection)
- tomDecomp (time-domain decomposition)
- INLSine (INL from sine wave)
- jitter_load (jitter analysis with deterministic data)
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

    # List of unit tests to run (in alphabetical order)
    unit_tests = [
        "test_errAutoCorrelation.py",
        "test_errEnvelopeSpectrum.py",
        "test_errHistSine.py",
        "test_errPDF.py",
        "test_errSpectrum.py",
        "test_INLSine.py",
        "test_jitter_load.py",
        "test_specPlot.py",
        "test_specPlotPhase.py",
        "test_tomDecomp.py",
    ]

    # Base directory for unit tests
    unit_test_dir = Path("tests/unit")

    results = []
    for test_file in unit_tests:
        test_path = unit_test_dir / test_file
        print(f"Running: {test_file}")
        print("-" * 80)

        # Run without capturing output to show real-time progress
        result = subprocess.run(
            [sys.executable, str(test_path)],
            capture_output=False,  # Show output in real-time
            text=True
        )

        success = (result.returncode == 0)
        results.append((test_file, success))

        if success:
            print(f"[FINISHED] {test_file}")
        else:
            print(f"[FAIL] {test_file}")

        print()

    # Summary
    print("=" * 80)
    print("Summary - Aout Package Unit Tests")
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
    success = run_unit_tests_aout()
    sys.exit(0 if success else 1)
