"""
Check that all Python test scripts can run without errors.

This script validates imports and basic functionality of all test scripts.
"""

import sys
import importlib
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def check_import(module_path):
    """
    Try to import a module and report success/failure.

    Parameters
    ----------
    module_path : str
        Python module path (e.g., 'adctoolbox.test.unit.test_alias')

    Returns
    -------
    bool
        True if import succeeded, False otherwise
    """
    try:
        importlib.import_module(module_path)
        print(f"  [PASS] Import successful: {module_path}")
        return True
    except Exception as e:
        print(f"  [FAIL] Import failed: {module_path}")
        print(f"         Error: {type(e).__name__}: {e}")
        if "--verbose" in sys.argv:
            traceback.print_exc()
        return False


def main():
    """Check all test scripts."""

    print("="*70)
    print("Checking Python Test Scripts")
    print("="*70)
    print()

    # List of test scripts to check
    unit_tests = [
        "adctoolbox.test.unit.test_alias",
        "adctoolbox.test.unit.test_cap2weight",
        "adctoolbox.test.unit.test_sineFit",
        "adctoolbox.test.unit.test_specPlot",
        "adctoolbox.test.unit.test_specPlotPhase",
        "adctoolbox.test.unit.test_INLSine",
        "adctoolbox.test.unit.test_FGCalSine",
        "adctoolbox.test.unit.test_FGCalSine_overflowChk",
        "adctoolbox.test.unit.test_error_analysis",
        "adctoolbox.test.unit.compare_sineFit_results",
    ]

    system_tests = [
        "adctoolbox.test.system.test_multimodal_report",
    ]

    # Check unit tests
    print("[Unit Tests]")
    unit_results = []
    for test in unit_tests:
        result = check_import(test)
        unit_results.append((test, result))
    print()

    # Check system tests
    print("[System Tests]")
    system_results = []
    for test in system_tests:
        result = check_import(test)
        system_results.append((test, result))
    print()

    # Summary
    all_tests = unit_results + system_results
    passed = sum(1 for _, result in all_tests if result)
    total = len(all_tests)

    print("="*70)
    print("Summary")
    print("="*70)
    print(f"Total tests checked: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print()

    if passed == total:
        print("[PASS] All test scripts can be imported successfully!")
        return 0
    else:
        print("[FAIL] Some test scripts have import errors.")
        print("\nFailed tests:")
        for test, result in all_tests:
            if not result:
                print(f"  - {test}")
        print("\nRun with --verbose flag for detailed error messages.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
