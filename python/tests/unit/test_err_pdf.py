"""test_err_pdf.py - Unit test for errPDF function

Tests the errPDF function with sinewave error data.

Output structure:
  test_output/<data_set_name>/test_errPDF/
      mu_python.csv           - Mean value
      sigma_python.csv        - Standard deviation
      KL_divergence_python.csv - KL divergence metric
      x_python.csv            - Histogram bin centers
      fx_python.csv           - Histogram values
      gauss_pdf_python.csv    - Gaussian PDF values
      errPDF_python.png       - PDF plot
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

from adctoolbox.common import sine_fit
from adctoolbox.aout import err_pdf
from save_variable import save_variable

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
            print(f"[{k}/{len(files_list)}] [test_errPDF] [ERROR] File not found: {current_filename}")
            continue

        # Read data
        read_data = np.loadtxt(data_file_path, delimiter=',', ndmin=1)

        # Extract dataset name
        dataset_name = data_file_path.stem

        # Create output subfolder
        sub_folder = output_dir / dataset_name / 'test_errPDF'
        sub_folder.mkdir(parents=True, exist_ok=True)

        # Compute error data using sineFit
        data_fit, freq_est, mag, dc, phi = sine_fit(read_data)
        err_data = read_data - data_fit

        # Run errPDF
        plt.figure(figsize=(12, 8))
        noise_lsb, mu, sigma, KL_divergence, x, fx, gauss_pdf = err_pdf(
            err_data,
            Resolution=12,
            FullScale=np.max(read_data) - np.min(read_data)
        )
        plt.title(f'errPDF: {dataset_name}')

        # Save plot
        plot_path = sub_folder / 'errPDF_python.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Save each variable to separate CSV (matching MATLAB format)
        save_variable(sub_folder, mu, 'mu')
        save_variable(sub_folder, sigma, 'sigma')
        save_variable(sub_folder, KL_divergence, 'KL_divergence')
        save_variable(sub_folder, x, 'x')
        save_variable(sub_folder, fx, 'fx')
        save_variable(sub_folder, gauss_pdf, 'gauss_pdf')

        # Print one-line progress
        print(f"[{k}/{len(files_list)}] [test_errPDF] [KL_div={KL_divergence:.4f}] from {current_filename}")

    print("[test_errPDF complete]")

    # Close any remaining figures
    plt.close('all')


if __name__ == "__main__":
    main()
