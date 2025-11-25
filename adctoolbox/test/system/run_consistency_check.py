"""run_consistency_check.py - Master script for MATLAB vs Python consistency checking

Runs all tests and generates comprehensive comparison reports.

Usage:
    python adctoolbox/test/unit/run_consistency_check.py
"""

import sys
import os
import subprocess
import time
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report status."""
    print()
    print("=" * 80)
    print(f"{description}")
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}")
    print()

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start_time

    if result.returncode == 0:
        status = "✅ SUCCESS"
    else:
        status = "⚠️  COMPLETED WITH WARNINGS"

    print()
    print(f"Status: {status}")
    print(f"Time: {elapsed:.1f}s")
    print()

    return result.returncode


def main():
    """Run complete consistency check workflow."""
    # Find project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parents[2]  # adctoolbox/test/unit -> adctoolbox -> ADCToolbox

    # Change to project root directory
    original_dir = os.getcwd()
    os.chdir(project_root)

    print("=" * 80)
    print("MATLAB vs Python Consistency Check")
    print("ADCToolbox - test_specPlot Function")
    print("=" * 80)
    print()
    print(f"Working directory: {project_root}")
    print()

    # Check if we're in the right directory
    if not os.path.exists("test_data"):
        print("ERROR: test_data directory not found.")
        print(f"Current directory: {project_root}")
        print("Expected directory structure:")
        print("  ADCToolbox/")
        print("    ├── test_data/")
        print("    ├── adctoolbox/")
        print("    └── ...")
        os.chdir(original_dir)
        return 1

    # Check if MATLAB results exist
    matlab_results = list(Path("test_output").rglob("metrics_matlab.csv"))
    if not matlab_results:
        print("⚠️  WARNING: No MATLAB results found in test_output/")
        print()
        print("Please run the MATLAB test first:")
        print("  1. Open MATLAB")
        print("  2. cd D:\\ADCToolbox")
        print("  3. Run: matlab/test/unit/test_specPlot.m")
        print()
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            os.chdir(original_dir)
            return 1
    else:
        print(f"✅ Found {len(matlab_results)} MATLAB result files")

    # Step 1: Run Python tests
    print()
    print("STEP 1: Running Python test_specPlot...")
    print()
    returncode = run_command(
        [sys.executable, "adctoolbox/test/unit/test_specPlot.py"],
        "Running Python test_specPlot"
    )

    # Step 2: Analyze comparisons
    print()
    print("STEP 2: Analyzing MATLAB vs Python comparisons...")
    print()
    run_command(
        [sys.executable, "adctoolbox/test/unit/analyze_comparison.py"],
        "Analyzing Consistency"
    )

    # Step 3: Create visual comparisons
    print()
    print("STEP 3: Creating visual comparisons...")
    print()
    run_command(
        [sys.executable, "adctoolbox/test/unit/visual_comparison.py"],
        "Creating Visual Comparisons"
    )

    # Final summary
    print()
    print("=" * 80)
    print("Consistency Check Complete!")
    print("=" * 80)
    print()
    print("Generated Reports:")
    print("  1. test_output/MATLAB_vs_Python_Comparison_Report.md")
    print("  2. test_output/test_specPlot_summary.csv")
    print()
    print("Output Structure:")
    print("  test_output/")
    print("    ├── <dataset_name>/")
    print("    │   └── test_specPlot/")
    print("    │       ├── spectrum_matlab.png")
    print("    │       ├── spectrum_python.png")
    print("    │       ├── metrics_matlab.csv")
    print("    │       ├── metrics_python.csv")
    print("    │       ├── comparison.csv")
    print("    │       └── comparison_visual.png")
    print("    ├── test_specPlot_summary.csv")
    print("    └── MATLAB_vs_Python_Comparison_Report.md")
    print()

    # Check if report exists
    if os.path.exists("test_output/MATLAB_vs_Python_Comparison_Report.md"):
        print("To view the detailed report:")
        print("  - Open: test_output/MATLAB_vs_Python_Comparison_Report.md")
        print("  - Or run: type test_output\\MATLAB_vs_Python_Comparison_Report.md")
    print()

    # Restore original directory
    os.chdir(original_dir)

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
