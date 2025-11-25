"""test_errHistSine.py - Unit test for errHistSine function

Tests the errHistSine function with sinewave datasets.

Output structure:
  test_output/<data_set_name>/test_errHistSine/
      metrics_python.csv          - anoi, pnoi
      phase_histogram_python.csv  - phase_code, emean, erms
      errHistSine_phase_python.png
      errHistSine_code_python.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

from adctoolbox.common import sine_fit
from adctoolbox.aout import errHistSine

# Get project root directory
project_root = Path(__file__).parent.parent.parent


def main():
    """Main test function."""
    input_dir = project_root / "test_data"
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
        emean, erms, phase_code, anoi, pnoi, err, xx = errHistSine(
            read_data, bin=360, fin=freq, disp=1, mode=0
        )

        # Current figure is the phase plot
        plt.gcf().suptitle(f'errHistSine (phase): {dataset_name}')

        # Save phase plot
        plot_path = sub_folder / 'errHistSine_phase_python.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Save metrics
        metrics_df = pd.DataFrame([{
            'anoi': anoi,
            'pnoi': pnoi
        }])
        metrics_path = sub_folder / 'metrics_python.csv'
        metrics_df.to_csv(metrics_path, index=False)

        # Save histogram data
        hist_df = pd.DataFrame({
            'phase_code': phase_code,
            'emean': emean,
            'erms': erms
        })
        hist_path = sub_folder / 'phase_histogram_python.csv'
        hist_df.to_csv(hist_path, index=False)

        # Run errHistSine - Code mode (mode=1)
        emean_code, erms_code, code_axis, _, _, _, _ = errHistSine(
            read_data, bin=256, fin=freq, disp=1, mode=1
        )

        # Current figure is the code plot
        plt.gcf().suptitle(f'errHistSine (code): {dataset_name}')

        # Save code plot
        plot_path = sub_folder / 'errHistSine_code_python.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Save code histogram data
        code_hist_df = pd.DataFrame({
            'code': code_axis,
            'emean': emean_code,
            'erms': erms_code
        })
        code_hist_path = sub_folder / 'code_histogram_python.csv'
        code_hist_df.to_csv(code_hist_path, index=False)

        # Print one-line progress
        print(f"[{k}/{len(files_list)}] [test_errHistSine] [pnoi={pnoi:.6f}rad] from {current_filename}")

    print("[test_errHistSine complete]")

    # Close any remaining figures
    plt.close('all')


if __name__ == "__main__":
    main()
