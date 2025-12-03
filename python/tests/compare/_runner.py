import pytest
from datetime import datetime
from tests.compare._csv_comparator import CSVComparator
from tests.compare._name_mapping import get_python_folder
from tests._utils import discover_test_datasets, discover_test_variables


def run_comparison_suite(project_root, matlab_test_name,
                         ref_folder="reference_output",
                         out_folder="test_output",
                         structure="nested",
                         enable_logging=False,
                         log_lines_output=None):
    """
    Generic runner for comparing MATLAB vs Python test results.

    Args:
        structure (str):
            - "nested": root / dataset / test_name (Default, e.g. sineFit)
            - "flat":   root / test_name (Simple, e.g. test_basic)
        enable_logging (bool): If True, creates individual log file for this test
        log_lines_output (list): If provided, appends log lines to this list instead of file
    """
    # 1. Setup
    ref_root = project_root / ref_folder  # Reference Data Root
    gen_root = project_root / out_folder  # Generated Data Root
    python_test_name = get_python_folder(matlab_test_name)

    # Setup logging
    if log_lines_output is None:
        log_lines = []
    else:
        log_lines = log_lines_output

    def log(msg):
        """Print and log message"""
        print(msg)
        log_lines.append(msg)

    log(f"[Comparison Test: {python_test_name}]")
    log(f"[MATLAB Test Name: {matlab_test_name}]")
    log(f"[Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
    log(f"[Mode: {structure.upper()}]")
    log(f"[Threshold: {CSVComparator.THRESHOLD:.2e}]")

    # -------------------------------------------------------
    # 2. Logic Branching: Determine Datasets & Paths
    # -------------------------------------------------------
    comparison_targets = []  # List of (display_name, mat_dir, py_dir)

    if structure == "nested":
        # Scenario: root / dataset / test_name
        datasets = discover_test_datasets(ref_root, subfolder=matlab_test_name)
        if not datasets:
            log(f"[WARNING] No datasets found in {ref_root} containing '{matlab_test_name}'")
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
            log(f"[WARNING] Flat reference folder not found: {mat_dir}")
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
        log(f"\n[{py_dir}] <-> [{mat_dir}]")

        # Check Directory (Python-Centric: skip if Python didn't generate it)
        if not py_dir.exists():
            log(f"  -> [SKIP] Python directory not found (dataset not processed)")
            continue

        # Discover Variables (Python-Centric: scan what Python generated)
        python_csv_files = sorted(py_dir.glob("*_python.csv"))
        if not python_csv_files:
            log(f"  -> [SKIP] No Python output files found")
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
                log(f"  [{var:<{max_len}}] -> [SKIP] No MATLAB reference")
                continue

            # Compare
            result = comparator.compare_pair(py_csv, mat_csv)
            status, diff = result['status'], result['max_diff_abs']

            msg = f"  [{var:<{max_len}}]: [Diff: {diff:.2e}] -> [{status:<8}]"

            if status == "PERFECT":
                log(msg)
            else:
                log(f"{msg} <--- FAIL")
                failures.append(
                    f"{display_name}/{var}: {status} (Diff: {diff:.2e})")

    # 4. Summary
    log("-" * 60)
    log(f"[SUMMARY] Verified {total_checks} files.")

    if failures:
        log("\n[FAILURES DETAILS]")
        log("\n".join(f"  x {f}" for f in failures))

    # Save log file if enabled and not using external log
    if enable_logging and log_lines_output is None:
        log_dir = project_root / "test_comparison_logs"
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{python_test_name}_{timestamp}.txt"
        log(f"\n[Log saved to: {log_file}]")
        log_file.write_text('\n'.join(log_lines), encoding='utf-8')

    if failures:
        raise AssertionError(f"Comparison failed for {len(failures)} item(s). check logs for details.")