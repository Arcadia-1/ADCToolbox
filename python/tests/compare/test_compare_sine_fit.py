"""
Pytest System Test - Compare MATLAB vs Python sineFit Outputs
"""

from pathlib import Path
from .csv_comparator import CSVComparator


def test_compare_sine_fit(project_root):
    """
    Compare Python sineFit outputs against MATLAB golden reference.
    Tests freq, mag, dc, phi, and data_fit outputs.
    """

    datasets = [
        "sinewave_jitter_400fs",
        "sinewave_noise_270uV",
    ]

    python_output_dir = project_root / "test_output"
    matlab_reference_dir = project_root / "test_reference"

    print(f"[INFO] Python output: [{python_output_dir}]")
    print(f"[INFO] MATLAB reference: [{matlab_reference_dir}]")

    comparator = CSVComparator()
    failures = []

    variables = ['freq', 'mag', 'dc', 'phi', 'data_fit']

    for dataset in datasets:
        print(f"\n[COMPARE] Dataset: {dataset}")

        python_dir = python_output_dir / dataset / 'test_sineFit'
        matlab_dir = matlab_reference_dir / dataset / 'test_sineFit'

        if not python_dir.exists():
            failures.append(f"{dataset}: Missing Python output directory")
            print(f"  -> [MISSING] Python directory not found: {python_dir}")
            continue

        if not matlab_dir.exists():
            failures.append(f"{dataset}: Missing MATLAB reference directory")
            print(f"  -> [MISSING] MATLAB directory not found: {matlab_dir}")
            continue

        dataset_passed = True

        for var in variables:
            py_csv = python_dir / f"{var}_python.csv"
            mat_csv = matlab_dir / f"{var}_matlab.csv"

            if not py_csv.exists():
                failures.append(f"{dataset}/{var}: Missing Python file")
                print(f"  -> [MISSING] {var}_python.csv")
                dataset_passed = False
                continue

            if not mat_csv.exists():
                failures.append(f"{dataset}/{var}: Missing MATLAB file")
                print(f"  -> [MISSING] {var}_matlab.csv")
                dataset_passed = False
                continue

            result = comparator.compare_pair(py_csv, mat_csv)
            status = result["status"]
            maxdiff = result.get("max_diff_pct", None)

            if status not in ["PERFECT", "EXCELLENT", "GOOD"]:
                failures.append(f"{dataset}/{var}: {status}")
                dataset_passed = False
                print(f"  -> [{var}] {status}")
                if maxdiff is not None:
                    print(f"     Max Diff: {maxdiff:.6f}%")
            else:
                print(f"  -> [{var}] {status}")

        if dataset_passed:
            print(f"  -> [PASS] All variables matched")
        else:
            print(f"  -> [FAIL] Some variables failed")

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total datasets: {len(datasets)}")
    print(f"Failures: {len(failures)}")

    if failures:
        print("\nFailed comparisons:")
        for f in failures:
            print(f"  - {f}")
        assert False, f"{len(failures)} comparisons failed"
    else:
        print("\n[PASS] All sineFit comparisons passed!")