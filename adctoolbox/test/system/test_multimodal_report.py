"""
Test multimodal_report.py - Multi-Modal Error Signatures Report Generator

Tests generation of 6-figure error analysis report from ADC data using existing
reference data files in tests/reference_data directory:
(a) Spectrum
(b) Time-domain error
(c) Phase-domain error (polar)
(d) Code overflow
(e) Error histogram by phase
(f) Error histogram by code
"""

import numpy as np
import os
import sys

# Add SpecMind directory to path (so ADC_Toolbox_Python can be imported as a package)
# This allows running the test from any directory
current_file = os.path.abspath(__file__)
tests_dir = os.path.dirname(current_file)  # tests/
adc_toolbox_dir = os.path.dirname(tests_dir)  # ADC_Toolbox_Python/
specmind_dir = os.path.dirname(adc_toolbox_dir)  # SpecMind/
sys.path.insert(0, specmind_dir)

from ADC_Toolbox_Python.multimodal_report import generate_multimodal_report


def test_basic_files():
    """Test multimodal report on basic reference data files"""

    print("\n" + "=" * 70)
    print("TEST 1: Basic reference data files")
    print("=" * 70)

    # Reference data directory
    ref_data_dir = os.path.join(
        os.path.dirname(__file__),
        'reference_data'
    )

    # Test files with known characteristics
    test_files = [
        ('Sine_wave_13_69_bit.csv', 12, 'Clean sine wave'),
        ('Sine_wave_10_70_bit_nonlinearity.csv', 10, 'Sine with nonlinearity'),
    ]

    results = []

    for filename, num_bits, description in test_files:
        filepath = os.path.join(ref_data_dir, filename)

        if not os.path.exists(filepath):
            print(f"\n[SKIP] {filename} - file not found")
            continue

        print(f"\n--- {description} ({filename}) ---")

        # Load data
        data = np.loadtxt(filepath, delimiter=',').flatten()
        print(f"  Loaded {len(data)} samples")

        # Output directory
        case_name = filename.replace('.csv', '')
        output_dir = os.path.join(
            os.path.dirname(__file__),
            'output',
            'multimodal_' + case_name
        )

        try:
            # Generate report (auto-detect frequency)
            output_paths = generate_multimodal_report(
                data=data,
                fs=1.0,  # Normalized
                num_bits=num_bits,
                fin=None,  # Auto-detect
                output_dir=output_dir
            )

            # Verify all 6 figures exist
            expected_files = [
                'fig_a_spectrum.png',
                'fig_b_time_error.png',
                'fig_c_phase_error.png',
                'fig_d_code_overflow.png',
                'fig_e_error_hist_phase.png',
                'fig_f_error_hist_code.png'
            ]

            all_exist = all(
                os.path.exists(os.path.join(output_dir, f))
                for f in expected_files
            )

            if all_exist:
                print(f"  [PASS] All 6 figures generated")
                results.append((filename, True))
            else:
                print(f"  [FAIL] Some figures missing")
                results.append((filename, False))

        except Exception as e:
            print(f"  [FAIL] Error: {e}")
            import traceback
            traceback.print_exc()
            results.append((filename, False))

    # Summary
    print("\nSummary:")
    passed_count = sum(1 for _, passed in results if passed)
    print(f"  {passed_count}/{len(results)} files passed")

    all_passed = all(r[1] for r in results)
    return all_passed


def test_jitter_files():
    """Test multimodal report on jitter data"""

    print("\n" + "=" * 70)
    print("TEST 2: Jitter impairment files")
    print("=" * 70)

    ref_data_dir = os.path.join(
        os.path.dirname(__file__),
        'reference_data'
    )

    # Jitter files
    test_files = [
        ('jitter_0P0002.csv', 12, 'Low jitter (0.0002)'),
        ('jitter_0P001.csv', 12, 'Medium jitter (0.001)'),
        ('jitter_0P002.csv', 12, 'High jitter (0.002)'),
    ]

    results = []

    for filename, num_bits, description in test_files:
        filepath = os.path.join(ref_data_dir, filename)

        if not os.path.exists(filepath):
            print(f"\n[SKIP] {filename} - file not found")
            continue

        print(f"\n--- {description} ---")

        # Load data
        data = np.loadtxt(filepath, delimiter=',').flatten()

        case_name = filename.replace('.csv', '')
        output_dir = os.path.join(
            os.path.dirname(__file__),
            'output',
            'multimodal_' + case_name
        )

        try:
            output_paths = generate_multimodal_report(
                data=data,
                fs=1.0,
                num_bits=num_bits,
                fin=None,
                output_dir=output_dir
            )

            all_exist = len(output_paths) == 6 and all(
                os.path.exists(p) for p in output_paths
            )

            status = "[PASS]" if all_exist else "[FAIL]"
            print(f"  {status}")
            results.append((filename, all_exist))

        except Exception as e:
            print(f"  [FAIL] Error: {e}")
            results.append((filename, False))

    print(f"\nSummary: {sum(1 for _, p in results if p)}/{len(results)} passed")
    return all(r[1] for r in results)


def test_gain_error_files():
    """Test multimodal report on gain error data"""

    print("\n" + "=" * 70)
    print("TEST 3: Gain error files")
    print("=" * 70)

    ref_data_dir = os.path.join(
        os.path.dirname(__file__),
        'reference_data'
    )

    # Gain error files
    test_files = [
        ('gain_error_0P95.csv', 12, 'Gain 0.95'),
        ('gain_error_1P05.csv', 12, 'Gain 1.05'),
    ]

    results = []

    for filename, num_bits, description in test_files:
        filepath = os.path.join(ref_data_dir, filename)

        if not os.path.exists(filepath):
            print(f"\n[SKIP] {filename} - file not found")
            continue

        print(f"\n--- {description} ---")

        data = np.loadtxt(filepath, delimiter=',').flatten()

        case_name = filename.replace('.csv', '')
        output_dir = os.path.join(
            os.path.dirname(__file__),
            'output',
            'multimodal_' + case_name
        )

        try:
            output_paths = generate_multimodal_report(
                data=data,
                fs=1.0,
                num_bits=num_bits,
                fin=None,
                output_dir=output_dir
            )

            all_exist = len(output_paths) == 6 and all(
                os.path.exists(p) for p in output_paths
            )

            status = "[PASS]" if all_exist else "[FAIL]"
            print(f"  {status}")
            results.append((filename, all_exist))

        except Exception as e:
            print(f"  [FAIL] Error: {e}")
            results.append((filename, False))

    print(f"\nSummary: {sum(1 for _, p in results if p)}/{len(results)} passed")
    return all(r[1] for r in results)


def test_diagnosis_bench_format():
    """Test output to diagnosis bench folder structure"""

    print("\n" + "=" * 70)
    print("TEST 4: Diagnosis bench folder format")
    print("=" * 70)

    ref_data_dir = os.path.join(
        os.path.dirname(__file__),
        'reference_data'
    )

    filepath = os.path.join(ref_data_dir, 'jitter_0P001.csv')

    if not os.path.exists(filepath):
        print("  [SKIP] jitter_0P001.csv not found")
        return True

    data = np.loadtxt(filepath, delimiter=',').flatten()

    # Diagnosis bench style output directory
    # Format: ADC_Diagnosis_Bench/SAR/SAR_10bit_j500ps
    test_dir = os.path.join(
        os.path.dirname(__file__),
        'output',
        'diagnosis_bench_format',
        'SAR',
        'SAR_12bit_jitter_test'
    )

    print(f"\nTest diagnosis bench format:")
    print(f"  Output: {test_dir}")
    print(f"  (simulating ADC_Diagnosis_Bench/SAR/SAR_12bit_jitter_test)")

    try:
        output_paths = generate_multimodal_report(
            data=data,
            fs=1e6,
            num_bits=12,
            fin=None,
            output_dir=test_dir
        )

        all_exist = len(output_paths) == 6 and all(
            os.path.exists(p) for p in output_paths
        )

        if all_exist:
            print(f"\n[PASS] Diagnosis bench format works!")
            print(f"  6 figures saved to: {test_dir}")
            return True
        else:
            print("\n[FAIL] Some figures missing")
            return False

    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_clip_files():
    """Test multimodal report on clipping data"""

    print("\n" + "=" * 70)
    print("TEST 5: Clipping files")
    print("=" * 70)

    ref_data_dir = os.path.join(
        os.path.dirname(__file__),
        'reference_data'
    )

    # Clipping files
    test_files = [
        ('clip_0P06.csv', 12, 'Clip 0.06'),
        ('clip_0P07.csv', 12, 'Clip 0.07'),
    ]

    results = []

    for filename, num_bits, description in test_files:
        filepath = os.path.join(ref_data_dir, filename)

        if not os.path.exists(filepath):
            print(f"\n[SKIP] {filename} - file not found")
            continue

        print(f"\n--- {description} ---")

        data = np.loadtxt(filepath, delimiter=',').flatten()

        case_name = filename.replace('.csv', '')
        output_dir = os.path.join(
            os.path.dirname(__file__),
            'output',
            'multimodal_' + case_name
        )

        try:
            output_paths = generate_multimodal_report(
                data=data,
                fs=1.0,
                num_bits=num_bits,
                fin=None,
                output_dir=output_dir
            )

            all_exist = len(output_paths) == 6 and all(
                os.path.exists(p) for p in output_paths
            )

            status = "[PASS]" if all_exist else "[FAIL]"
            print(f"  {status}")
            results.append((filename, all_exist))

        except Exception as e:
            print(f"  [FAIL] Error: {e}")
            results.append((filename, False))

    print(f"\nSummary: {sum(1 for _, p in results if p)}/{len(results)} passed")
    return all(r[1] for r in results)


def run_all_tests():
    """Run all multimodal_report tests"""

    print("\n" + "=" * 70)
    print("MULTIMODAL REPORT - TEST SUITE")
    print("Testing with reference_data files")
    print("=" * 70)

    tests = [
        ("Basic reference files", test_basic_files),
        ("Jitter impairments", test_jitter_files),
        ("Gain errors", test_gain_error_files),
        ("Diagnosis bench format", test_diagnosis_bench_format),
        ("Clipping files", test_clip_files),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n[EXCEPTION] {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} - {test_name}")

    all_passed = all(r[1] for r in results)
    total = len(results)
    passed_count = sum(r[1] for r in results)

    print(f"\nTotal: {passed_count}/{total} tests passed")

    if all_passed:
        print("\n*** ALL TESTS PASSED! ***")
    else:
        print("\n*** SOME TESTS FAILED ***")

    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
