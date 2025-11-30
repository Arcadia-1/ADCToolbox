"""test_tom_decomp.py - Unit test for tomDecomp function

Tests the Thompson Decomposition function with sinewave datasets.

Output structure:
  test_output/<data_set_name>/test_tomDecomp/
      decomp_data_python.csv  - signal, error, indep, dep (first 1000 samples)
      metrics_python.csv      - phi, rms_error, rms_indep, rms_dep
      tomDecomp_python.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

from adctoolbox.common import find_fin
from adctoolbox.aout import tom_decomp
from save_variable import save_variable
from save_fig import save_fig

# Get project root directory
project_root = Path(__file__).resolve().parents[3]


def main():
    """Main test function."""
    input_dir = project_root / "dataset" / "aout"
    output_dir = project_root / "test_output"

    # Test datasets - leave empty to auto-search
    files_list = []

    # Auto-search if list is empty
    if not files_list:
        search_patterns = ['sinewave_*.csv']
        files_list = []
        for pattern in search_patterns:
            files_list.extend([f.name for f in input_dir.glob(pattern)])

    if not files_list:
        print(f"ERROR: No test files found in {input_dir}")
        return

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Test Loop
    for k, current_filename in enumerate(files_list, start=1):
        data_file_path = input_dir / current_filename

        if not data_file_path.is_file():
            print(f"[{k}/{len(files_list)}] [test_tomDecomp] [ERROR] File not found: {current_filename}")
            continue

        # Read data
        read_data = np.loadtxt(data_file_path, delimiter=',', ndmin=1)

        # Extract dataset name
        dataset_name = data_file_path.stem

        # Create output subfolder
        sub_folder = output_dir / dataset_name / 'test_tomDecomp'
        sub_folder.mkdir(parents=True, exist_ok=True)

        # Find input frequency
        re_fin = find_fin(read_data)

        # Run tomDecomp
        signal, error, indep, dep, phi = tom_decomp(read_data, re_fin, 10, 1)

        # Current figure is the decomposition plot
        plt.gcf().suptitle(f'tomDecomp: {dataset_name}')

        # Save plot
        save_fig(sub_folder, 'tomDecomp_python.png')

        # Calculate metrics
        rms_error = np.sqrt(np.mean(error**2))
        rms_indep = np.sqrt(np.mean(indep**2))
        rms_dep = np.sqrt(np.mean(dep**2))

        # Save each variable to separate CSV (matching MATLAB format)
        save_variable(sub_folder, signal, 'signal')
        save_variable(sub_folder, error, 'error')
        save_variable(sub_folder, indep, 'indep')
        save_variable(sub_folder, dep, 'dep')
        save_variable(sub_folder, phi, 'phi')
        save_variable(sub_folder, rms_error, 'rms_error')
        save_variable(sub_folder, rms_indep, 'rms_indep')
        save_variable(sub_folder, rms_dep, 'rms_dep')

        # Print one-line progress
        print(f"[{k}/{len(files_list)}] [test_tomDecomp] [rms_err={rms_error:.6f}] from {current_filename}")

    print("[test_tomDecomp complete]")

    # Close any remaining figures
    plt.close('all')


if __name__ == "__main__":
    main()
