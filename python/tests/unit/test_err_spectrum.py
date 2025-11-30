"""test_err_spectrum.py - Unit test for error spectrum analysis

Tests error spectrum plotting with sinewave error data.

Output structure:
  test_output/<data_set_name>/test_errSpectrum/
      errSpectrum_python.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

from adctoolbox.common import sine_fit
from adctoolbox.aout import spec_plot

# Get project root directory (two levels up from python/tests/unit)
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
            print(f"[{k}/{len(files_list)}] [test_errSpectrum] [ERROR] File not found: {current_filename}")
            continue

        # Read data
        read_data = np.loadtxt(data_file_path, delimiter=',', ndmin=1)

        # Extract dataset name
        dataset_name = data_file_path.stem

        # Create output subfolder
        sub_folder = output_dir / dataset_name / 'test_errSpectrum'
        sub_folder.mkdir(parents=True, exist_ok=True)

        # Compute error data using sineFit
        data_fit, freq_est, mag, dc, phi = sine_fit(read_data)
        err_data = read_data - data_fit

        # Run spec_plot on error data (label=0 means no labeling)
        plt.figure(figsize=(12, 8))
        spec_plot(err_data, label=0)
        plt.title(f'errSpectrum: {dataset_name}')

        # Save plot
        plot_path = sub_folder / 'errSpectrum_python.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Print one-line progress
        print(f"[{k}/{len(files_list)}] [test_errSpectrum] [OK] from {current_filename}")

    print("[test_errSpectrum complete]")

    # Close any remaining figures
    plt.close('all')


if __name__ == "__main__":
    main()
