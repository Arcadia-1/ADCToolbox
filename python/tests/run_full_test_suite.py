"""
Master test runner for ADCToolbox Python test suite.

This script orchestrates the complete test workflow:
1. Run unit tests to generate Python outputs
2. Run comparison tests to validate against MATLAB references

Usage:
    pytest run_full_test_suite.py -v
    pytest run_full_test_suite.py::test_full_suite_common -v
    pytest run_full_test_suite.py::test_full_suite_analog -v
    pytest run_full_test_suite.py::test_full_suite_digital -v
"""

import pytest
from pathlib import Path


@pytest.fixture
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent.parent


def test_full_suite_common(project_root):
    """
    Run full test suite for common tests:
    1. Generate outputs (unit tests)
    2. Validate against MATLAB (comparison tests)
    """
    print("\n" + "=" * 70)
    print("COMMON TEST SUITE")
    print("=" * 70)

    # Step 1: Generate Python outputs
    print("\n[STEP 1/2] Running unit tests to generate outputs...")
    print("-" * 70)

    unit_tests = [
        "tests/unit/test_basic.py",
        "tests/unit/test_sine_fit.py",
        "tests/unit/test_alias.py",
    ]

    for test in unit_tests:
        print(f"\nRunning: {test}")
        result = pytest.main(["-xvs", str(project_root / "python" / test)])
        if result != 0:
            raise AssertionError(f"Unit test failed: {test}")

    # Step 2: Run comparison tests
    print("\n[STEP 2/2] Running comparison tests...")
    print("-" * 70)

    comparison_tests = [
        "tests/compare/test_compare_basic.py",
        "tests/compare/test_compare_sine_fit.py",
    ]

    for test in comparison_tests:
        print(f"\nRunning: {test}")
        result = pytest.main(["-xvs", str(project_root / "python" / test)])
        if result != 0:
            raise AssertionError(f"Comparison test failed: {test}")

    print("\n" + "=" * 70)
    print("✓ COMMON TEST SUITE PASSED")
    print("=" * 70)


def test_full_suite_analog(project_root):
    """
    Run full test suite for analog output (aout) tests:
    1. Generate outputs (unit tests)
    2. Validate against MATLAB (comparison tests)
    """
    print("\n" + "=" * 70)
    print("ANALOG OUTPUT (AOUT) TEST SUITE")
    print("=" * 70)

    # Step 1: Generate Python outputs
    print("\n[STEP 1/2] Running unit tests to generate outputs...")
    print("-" * 70)

    unit_tests = [
        "tests/unit/test_spec_plot.py",
        "tests/unit/test_spec_plot_phase.py",
        "tests/unit/test_err_hist_sine_code.py",
        "tests/unit/test_err_hist_sine_phase.py",
        "tests/unit/test_tom_decomp.py",
        "tests/unit/test_err_pdf.py",
        "tests/unit/test_err_auto_correlation.py",
        "tests/unit/test_err_spectrum.py",
        "tests/unit/test_err_envelope_spectrum.py",
        "tests/unit/test_inl_sine.py",
    ]

    for test in unit_tests:
        test_path = project_root / "python" / test
        if test_path.exists():
            print(f"\nRunning: {test}")
            result = pytest.main(["-xvs", str(test_path)])
            if result != 0:
                raise AssertionError(f"Unit test failed: {test}")
        else:
            print(f"\n[SKIP] Test not found: {test}")

    # Step 2: Run comparison tests
    print("\n[STEP 2/2] Running comparison tests...")
    print("-" * 70)

    comparison_tests = [
        "tests/compare/test_compare_spec_plot.py",
        "tests/compare/test_compare_spec_plot_phase.py",
        "tests/compare/test_compare_err_hist_sine_code.py",
        "tests/compare/test_compare_err_hist_sine_phase.py",
    ]

    for test in comparison_tests:
        test_path = project_root / "python" / test
        if test_path.exists():
            print(f"\nRunning: {test}")
            result = pytest.main(["-xvs", str(test_path)])
            if result != 0:
                raise AssertionError(f"Comparison test failed: {test}")
        else:
            print(f"\n[SKIP] Test not found: {test}")

    print("\n" + "=" * 70)
    print("✓ ANALOG OUTPUT TEST SUITE PASSED")
    print("=" * 70)


def test_full_suite_digital(project_root):
    """
    Run full test suite for digital output (dout) tests:
    1. Generate outputs (unit tests)
    2. Validate against MATLAB (comparison tests)
    """
    print("\n" + "=" * 70)
    print("DIGITAL OUTPUT (DOUT) TEST SUITE")
    print("=" * 70)

    # Step 1: Generate Python outputs
    print("\n[STEP 1/2] Running unit tests to generate outputs...")
    print("-" * 70)

    unit_tests = [
        "tests/unit/test_bit_activity.py",
        "tests/unit/test_enob_bit_sweep.py",
        "tests/unit/test_fg_cal_sine.py",
        "tests/unit/test_weight_scaling.py",
    ]

    for test in unit_tests:
        test_path = project_root / "python" / test
        if test_path.exists():
            print(f"\nRunning: {test}")
            result = pytest.main(["-xvs", str(test_path)])
            if result != 0:
                raise AssertionError(f"Unit test failed: {test}")
        else:
            print(f"\n[SKIP] Test not found: {test}")

    # Step 2: Run comparison tests
    print("\n[STEP 2/2] Running comparison tests...")
    print("-" * 70)

    comparison_tests = [
        "tests/compare/test_compare_bit_activity.py",
        "tests/compare/test_compare_enob_bit_sweep.py",
        "tests/compare/test_compare_fg_cal_sine.py",
        "tests/compare/test_compare_weight_scaling.py",
        "tests/compare/test_compare_fg_cal_sine_overflow_chk.py",
    ]

    for test in comparison_tests:
        test_path = project_root / "python" / test
        if test_path.exists():
            print(f"\nRunning: {test}")
            result = pytest.main(["-xvs", str(test_path)])
            if result != 0:
                raise AssertionError(f"Comparison test failed: {test}")
        else:
            print(f"\n[SKIP] Test not found: {test}")

    print("\n" + "=" * 70)
    print("✓ DIGITAL OUTPUT TEST SUITE PASSED")
    print("=" * 70)


def test_full_suite_all(project_root):
    """
    Run the complete test suite (common + analog + digital).
    This is the master test that runs everything.
    """
    print("\n" + "=" * 70)
    print("MASTER TEST SUITE - RUNNING ALL TESTS")
    print("=" * 70)

    # Run all test suites
    test_full_suite_common(project_root)
    test_full_suite_analog(project_root)
    test_full_suite_digital(project_root)

    print("\n" + "=" * 70)
    print("✓✓✓ ALL TEST SUITES PASSED ✓✓✓")
    print("=" * 70)
