"""
Golden Reference Tests - Dual Validation System

This test suite performs TWO validations:
1. Regression Test: New Python output vs Python golden reference
2. Cross-Platform Parity: Python golden vs MATLAB golden reference

Both MATLAB and Python golden references are stored in test_reference/
and committed to git.
"""

import numpy as np
import json
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / 'python' / 'src'))

from adctoolbox.common import sine_fit
from adctoolbox.aout import spec_plot


def load_tolerance(dataset_name):
    """Load tolerance settings for a dataset"""
    tolerance_path = project_root / 'test_reference' / dataset_name / 'tolerance.json'
    with open(tolerance_path) as f:
        return json.load(f)


def load_csv(csv_path):
    """Load CSV, handling header rows"""
    if not csv_path.exists():
        return None

    try:
        return np.loadtxt(csv_path, delimiter=',', ndmin=1)
    except ValueError:
        # Has header row, skip it
        return np.loadtxt(csv_path, delimiter=',', ndmin=1, skiprows=1)


def compare_values(new_python, python_golden, matlab_golden, var_name, tolerance):
    """
    Dual validation:
    1. Compare new Python vs Python golden (regression)
    2. Verify Python golden vs MATLAB golden (parity)
    """
    atol = tolerance.get('atol', 0.01)
    rtol = tolerance.get('rtol', 1e-4)

    new_python_arr = np.atleast_1d(new_python)

    # Check 1: Regression test (new Python vs Python golden)
    if python_golden is not None:
        if np.allclose(new_python_arr, python_golden, atol=atol, rtol=rtol):
            max_diff = np.max(np.abs(new_python_arr - python_golden))
            print(f"  [PASS] {var_name} (regression): max_diff={max_diff:.6e}")
            regression_pass = True
        else:
            max_diff = np.max(np.abs(new_python_arr - python_golden))
            print(f"  [FAIL] {var_name} (regression): max_diff={max_diff:.6e} > tol={atol}")
            print(f"    New:    {new_python_arr[:3]}...")
            print(f"    Golden: {python_golden[:3]}...")
            regression_pass = False
    else:
        print(f"  [SKIP] {var_name} (regression): no Python golden reference")
        regression_pass = True

    # Check 2: Cross-platform parity (Python golden vs MATLAB golden)
    if python_golden is not None and matlab_golden is not None:
        if np.allclose(python_golden, matlab_golden, atol=atol, rtol=rtol):
            max_diff = np.max(np.abs(python_golden - matlab_golden))
            print(f"  [PASS] {var_name} (parity): Python matches MATLAB, diff={max_diff:.6e}")
            parity_pass = True
        else:
            max_diff = np.max(np.abs(python_golden - matlab_golden))
            print(f"  [FAIL] {var_name} (parity): Python != MATLAB, diff={max_diff:.6e}")
            print(f"    Python: {python_golden[:3]}...")
            print(f"    MATLAB: {matlab_golden[:3]}...")
            parity_pass = False
    else:
        print(f"  [SKIP] {var_name} (parity): missing reference")
        parity_pass = True

    return regression_pass and parity_pass


def test_sinewave_jitter_400fs_sineFit():
    """Test sine_fit with dual validation (regression + parity)"""
    print("\n=== test_sineFit (sinewave_jitter_400fs) ===")

    dataset_name = 'sinewave_jitter_400fs'

    # Load input
    input_path = project_root / 'dataset' / f'{dataset_name}.csv'
    data = np.loadtxt(input_path, delimiter=',')

    # Run Python function (new output)
    data_fit, freq, mag, dc, phi = sine_fit(data)

    # Load tolerance
    tolerance = load_tolerance(dataset_name)
    test_tol = tolerance['test_sineFit']

    # Load golden references
    ref_dir = project_root / 'test_reference' / dataset_name / 'test_sineFit'

    all_passed = True

    # Validate each output variable
    for var_name, new_value, csv_name in [
        ('freq', freq, 'freq'),
        ('mag', mag, 'mag'),
        ('dc', dc, 'dc'),
        ('phi', phi, 'phi'),
        ('data_fit', data_fit[:1000], 'data_fit')  # First 1000 samples
    ]:
        python_golden = load_csv(ref_dir / f'{csv_name}_python.csv')
        matlab_golden = load_csv(ref_dir / f'{csv_name}_matlab.csv')

        passed = compare_values(
            new_value,
            python_golden,
            matlab_golden,
            var_name,
            test_tol[csv_name]
        )
        all_passed &= passed

    assert all_passed, "Some validations failed"
    print("==> All sineFit validations passed")


def test_sinewave_jitter_400fs_specPlot():
    """Test spec_plot against golden reference"""
    print("\n=== test_specPlot (sinewave_jitter_400fs) ===")

    dataset_name = 'sinewave_jitter_400fs'

    # Load input
    input_path = project_root / 'dataset' / f'{dataset_name}.csv'
    data = np.loadtxt(input_path, delimiter=',')

    # Run Python function
    ENoB, SNDR, SFDR, SNR, THD, pwr, NF, _ = spec_plot(data, Fs=1024, isPlot=False)

    # Load tolerance
    tolerance = load_tolerance(dataset_name)
    test_tol = tolerance['test_specPlot']

    # Compare outputs
    ref_dir = project_root / 'test_reference' / dataset_name / 'test_specPlot'

    all_passed = True
    all_passed &= compare_csv(ENoB, ref_dir / 'ENoB_matlab.csv', test_tol['ENoB'])
    all_passed &= compare_csv(SNDR, ref_dir / 'SNDR_matlab.csv', test_tol['SNDR'])
    all_passed &= compare_csv(SFDR, ref_dir / 'SFDR_matlab.csv', test_tol['SFDR'])
    all_passed &= compare_csv(SNR, ref_dir / 'SNR_matlab.csv', test_tol['SNR'])
    all_passed &= compare_csv(THD, ref_dir / 'THD_matlab.csv', test_tol['THD'])
    all_passed &= compare_csv(pwr, ref_dir / 'pwr_matlab.csv', test_tol['pwr'])
    all_passed &= compare_csv(NF, ref_dir / 'NF_matlab.csv', test_tol['NF'])

    assert all_passed, "Some outputs didn't match MATLAB golden reference"
    print("✓ All specPlot outputs match MATLAB")


if __name__ == '__main__':
    """Run tests manually"""
    print("="*60)
    print("Golden Reference Validation Tests")
    print("Comparing Python outputs vs MATLAB reference")
    print("="*60)

    try:
        test_sinewave_jitter_400fs_sineFit()
        test_sinewave_jitter_400fs_specPlot()

        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED - Python matches MATLAB")
        print("="*60)

    except AssertionError as e:
        print("\n" + "="*60)
        print(f"✗ TEST FAILED: {e}")
        print("="*60)
        sys.exit(1)
    except Exception as e:
        print("\n" + "="*60)
        print(f"✗ ERROR: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        sys.exit(1)
