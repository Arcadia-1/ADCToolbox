"""test_err_hist_sine.py - Unit test for errHistSine function

Tests the errHistSine function with sinewave datasets.

Output structure:
  test_output/<data_set_name>/test_errHistSine/
      anoi_python.csv             - Amplitude noise
      pnoi_python.csv             - Phase noise
      phase_code_python.csv       - Phase codes
      emean_python.csv            - Error mean (phase)
      erms_python.csv             - Error RMS (phase)
      code_axis_python.csv        - Code axis values
      emean_code_python.csv       - Error mean (code)
      erms_code_python.csv        - Error RMS (code)
      errHistSine_phase_python.png
      errHistSine_code_python.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

from adctoolbox.common import sine_fit
from adctoolbox.aout import err_hist_sine
from save_variable import save_variable
from save_fig import save_fig

# Get project root directory (two levels up from python/tests/unit)
project_root = Path(__file__).resolve().parents[3]

def main():
    """Main test function."""
    input_dir = project_root / "dataset"
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
            print(f"[{k}/{len(files_list)}] [test_errHistSine] [ERROR] File not found: {current_filename}")
            continue

        # Read data
        read_data = np.loadtxt(data_file_path, delimiter=',', ndmin=1)

        # Extract dataset name
        dataset_name = data_file_path.stem

        # Create output subfolder
        sub_folder = output_dir / dataset_name / 'test_errHistSine'
        sub_folder.mkdir(parents=True, exist_ok=True)

        # Get frequency estimate
        data_fit, freq, mag, dc, phi = sine_fit(read_data)

        # Run errHistSine - Phase mode (mode=0)
        emean, erms, phase_code, anoi, pnoi, err, xx = err_hist_sine(
            read_data, bin=360, fin=freq, disp=1, mode=0
        )

        # Current figure is the phase plot
        plt.gcf().suptitle(f'errHistSine (phase): {dataset_name}')

        # Save phase plot and variables
        save_fig(sub_folder, 'errHistSine_phase_python.png')
        save_variable(sub_folder, anoi, 'anoi')
        save_variable(sub_folder, pnoi, 'pnoi')
        save_variable(sub_folder, phase_code, 'phase_code')
        save_variable(sub_folder, emean, 'emean')
        save_variable(sub_folder, erms, 'erms')

        # Run errHistSine - Code mode (mode=1)
        emean_code, erms_code, code_axis, _, _, _, _ = err_hist_sine(
            read_data, bin=256, fin=freq, disp=1, mode=1
        )

        # Current figure is the code plot
        plt.gcf().suptitle(f'errHistSine (code): {dataset_name}')

        # Save code plot and variables
        save_fig(sub_folder, 'errHistSine_code_python.png')
        save_variable(sub_folder, code_axis, 'code_axis')
        save_variable(sub_folder, emean_code, 'emean_code')
        save_variable(sub_folder, erms_code, 'erms_code')

        # Print one-line progress
        print(f"[{k}/{len(files_list)}] [test_errHistSine] [pnoi={pnoi:.6f}rad] from {current_filename}")

    print("[test_errHistSine complete]")

    # Close any remaining figures
    plt.close('all')


if __name__ == "__main__":
    main()
