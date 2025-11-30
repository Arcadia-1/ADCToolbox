import pytest
from .csv_comparator import CSVComparator

def test_compare_all_types(project_root):
    """
    Validates Vector, Matrix, and Scalar generation against MATLAB reference.
    Strict equality check (Absolute Error only).
    """
    base_dir = project_root / 'test_output' / 'test_basic'
    
    # 1. Define the targets
    targets = [
        "sinewave",    # Vector
        "test_matrix", # Matrix
        "test_scalar"  # Scalar
    ]

    comparator = CSVComparator()
    failures = [] 

    # 2. Iterate
    for name in targets:
        py_csv = base_dir / f"{name}_python.csv"
        mat_csv = base_dir / f"{name}_matlab.csv"

        print()
        print(f"Python: [{py_csv}]\nMatlab: [{mat_csv}]")

        # Check existence
        if not py_csv.exists(): 
            print(f"-> [MISSING] Python file not found")
            failures.append(f"{name}: Missing Python file")
            continue
        if not mat_csv.exists():
            print(f"-> [MISSING] MATLAB file not found")
            failures.append(f"{name}: Missing MATLAB file")
            continue

        # Execute Comparison
        result = comparator.compare_pair(py_csv, mat_csv)

        # Print Result
        if result['status'] != 'ERROR':
            diff_str = f"{result['max_diff_abs']:.2e}"
            shape_str = str(result.get('shape', 'N/A'))
            # Result indented to align with paths
            print(f"  -> [RESULT] {result['status']} | Diff: {diff_str} | Shape: {shape_str}")
        else:
            print(f"  -> [ERROR] Msg: {result['msg']}")
            failures.append(f"{name}: Crashed ({result['msg']})")

        # Check failure
        if result['status'] == 'FAIL':
            failures.append(f"{name}: Diff {result['max_diff_abs']:.2e} > Limit")

    # 3. Final Assertion
    if failures:
        pytest.fail(f"Validation failed for {len(failures)} item(s):\n" + "\n".join(failures))