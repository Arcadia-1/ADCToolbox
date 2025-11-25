"""test_errAutoCorrelation.py - Unit test for errAutoCorrelation function

Tests the errAutoCorrelation function with sinewave error data.

Output structure:
  test_output/<data_set_name>/test_errAutoCorrelation/
      acf_data_python.csv     - lags, acf
      errACF_python.png       - ACF plot
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

from adctoolbox.common import sine_fit
from adctoolbox.aout import errAutoCorrelation

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
            print(f"[{k}/{len(files_list)}] [test_errAutoCorrelation] [ERROR] File not found: {current_filename}")
            continue

        # Read data
        read_data = np.loadtxt(data_file_path, delimiter=',', ndmin=1)

        # Extract dataset name
        dataset_name = data_file_path.stem

        # Create output subfolder
        sub_folder = output_dir / dataset_name / 'test_errAutoCorrelation'
        sub_folder.mkdir(parents=True, exist_ok=True)

        # Compute error data using sineFit
        data_fit, freq_est, mag, dc, phi = sine_fit(read_data)
        err_data = read_data - data_fit

        # Run errAutoCorrelation
        plt.figure(figsize=(12, 8))
        acf, lags = errAutoCorrelation(err_data, MaxLag=200)
        plt.title(f'errAutoCorrelation: {dataset_name}')

        # Save plot
        plot_path = sub_folder / 'errACF_python.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Save ACF data to CSV
        acf_df = pd.DataFrame({
            'lags': lags,
            'acf': acf
        })
        acf_path = sub_folder / 'acf_data_python.csv'
        acf_df.to_csv(acf_path, index=False)

        # Get ACF at lag 0 for display
        acf_0 = acf[np.where(lags == 0)[0][0]] if 0 in lags else acf[0]

        # Print one-line progress
        print(f"[{k}/{len(files_list)}] [test_errAutoCorrelation] [ACF(0)={acf_0:.4f}] from {current_filename}")

    print("[test_errAutoCorrelation complete]")

    # Close any remaining figures
    plt.close('all')


if __name__ == "__main__":
    main()
