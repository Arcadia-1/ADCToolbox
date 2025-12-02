import pytest
from python.tests.compare._csv_comparator import CSVComparator
from python.tests.compare._name_mapping import get_python_folder
from tests._utils import discover_test_datasets, discover_test_variables


def run_comparison_suite(project_root, matlab_test_name,
                         ref_folder="test_reference",
                         out_folder="test_output",
                         structure="nested"):
    """
    Generic runner for comparing MATLAB vs Python test results.

    Args:
        structure (str): 
            - "nested": root / dataset / test_name (Default, e.g. sineFit)
            - "flat":   root / test_name (Simple, e.g. test_basic)
    """
    # 1. Setup
    ref_root = project_root / ref_folder  # Reference Data Root
    gen_root = project_root / out_folder  # Generated Data Root
    python_test_name = get_python_folder(matlab_test_name)

    print(f"[Mode: {structure.upper()}]")

    # -------------------------------------------------------
    # 2. Logic Branching: Determine Datasets & Paths
    # -------------------------------------------------------
    comparison_targets = []  # List of (display_name, mat_dir, py_dir)

    if structure == "nested":
        # Scenario: root / dataset / test_name
        datasets = discover_test_datasets(ref_root, subfolder=matlab_test_name)
        if not datasets:
            print(
                f"[WARNING] No datasets found in {ref_root} containing '{matlab_test_name}'")
            return

        for ds in datasets:
            mat_dir = ref_root / ds / matlab_test_name
            py_dir = gen_root / ds / python_test_name
            comparison_targets.append((ds, mat_dir, py_dir))

    elif structure == "flat":
        # Scenario: root / test_name (No dataset layer)
        # We treat the test name itself as the "dataset" for display purposes
        mat_dir = ref_root / matlab_test_name
        py_dir = gen_root / python_test_name

        # Check if the folder exists at all
        if not mat_dir.exists():
            print(f"[WARNING] Flat reference folder not found: {mat_dir}")
            return

        comparison_targets.append(("Single", mat_dir, py_dir))

    else:
        raise ValueError(f"Unknown structure mode: {structure}")

    # -------------------------------------------------------
    # 3. Execution Loop (Common Logic)
    # -------------------------------------------------------
    comparator = CSVComparator()
    failures = []
    total_checks = 0

    for display_name, mat_dir, py_dir in comparison_targets:
        print(f"\n[{py_dir}] <-> [{mat_dir}]")

        # Check Directory
        if not py_dir.exists():
            print(f"  -> [MISSING] Python directory: {py_dir.name}")
            failures.append(f"{display_name}: Missing Python directory")
            continue

        # Discover Variables (Python-Centric: scan what Python generated)
        python_csv_files = sorted(py_dir.glob("*_python.csv"))
        if not python_csv_files:
            print(f"  -> [SKIP] No Python output files found")
            continue

        # Extract variable names from Python files
        variables = [f.stem.replace("_python", "") for f in python_csv_files]

        max_len = max((len(v) for v in variables), default=8)
        max_len = max(max_len, 8)

        for var in variables:
            total_checks += 1
            py_csv = py_dir / f"{var}_python.csv"
            mat_csv = mat_dir / f"{var}_matlab.csv"

            # Check if MATLAB reference exists (Skip if no reference, not FAIL)
            if not mat_csv.exists():
                print(f"  [{var:<{max_len}}] -> [SKIP] No MATLAB reference")
                continue

            # Compare
            result = comparator.compare_pair(py_csv, mat_csv)
            status, diff = result['status'], result['max_diff_abs']

            msg = f"  [{var:<{max_len}}]: [Diff: {diff:.2e}] -> [{status:<8}]"

            if status == "PERFECT":
                print(msg)
            else:
                print(f"{msg} <--- FAIL")
                failures.append(
                    f"{display_name}/{var}: {status} (Diff: {diff:.2e})")

    # 4. Summary
    print("-" * 60)
    print(f"[SUMMARY] Verified {total_checks} files.")

    if failures:
        print("\n[FAILURES DETAILS]")
        print("\n".join(f"  x {f}" for f in failures))
        
        raise AssertionError(f"Comparison failed for {len(failures)} item(s). check logs for details.")