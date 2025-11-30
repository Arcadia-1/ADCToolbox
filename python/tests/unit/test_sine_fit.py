"""
test_sine_fit.py - Unit test for sineFit function

Tests the sine_fit function for sinewave fitting with proper MxN support.

Output structure:
    test_output/<data_set_name>/test_sineFit/
        freq_python.csv         - Frequency estimates
        mag_python.csv          - Magnitude estimates
        dc_python.csv           - DC offset estimates
        phi_python.csv          - Phase estimates
        data_fit_python.csv     - First 1000 samples of fitted data
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path

from adctoolbox.common import sine_fit
from save_variable import save_variable

# Get project root directory (two levels up from python/tests/unit)
project_root = Path(__file__).resolve().parents[3]

def test_sineFit():
    """Test sineFit function on multiple datasets."""

    # Configuration
    input_dir = project_root / "dataset" / "aout"
    output_dir = project_root / "test_output"

    # Test datasets - leave empty to auto-search
    files_list = [
        # "batch_sinewave_Nrun_2.csv"
    ]

    # Auto-search if list is empty
    if not files_list:
        search_patterns = ['sinewave_*.csv']
        files_list = []
        for pattern in search_patterns:
            files_list.extend([f.name for f in input_dir.glob(pattern)])
        print(f"Auto-discovered {len(files_list)} files matching patterns: {', '.join(search_patterns)}")

    if not files_list:
        raise ValueError(f"No test files found in {input_dir}")

    output_dir.mkdir(exist_ok=True)

    # Test Loop
    print("=== test_sine_fit.py ===")
    print(f"Testing sine_fit function with {len(files_list)} datasets...\n")

    for k, current_filename in enumerate(files_list, 1):
        data_file_path = input_dir / current_filename

        if not data_file_path.is_file():
            print(f"[{k}/{len(files_list)}] {current_filename} - NOT FOUND, skipping\n")
            continue

        print(f"[{k}/{len(files_list)}] {current_filename} - found")

        # Read data
        read_data = pd.read_csv(data_file_path, header=None).values

        # Auto-transpose if needed (samples should be in rows, not columns)
        if read_data.ndim == 2 and read_data.shape[0] < read_data.shape[1]:
            read_data = read_data.T
            print(f"  (Auto-transposed to {read_data.shape})")

        # Extract dataset name
        dataset_name = data_file_path.stem

        # Create output subfolder
        sub_folder = output_dir / dataset_name / 'test_sineFit'
        sub_folder.mkdir(parents=True, exist_ok=True)

        # Run sineFit
        data_fit, freq, mag, dc, phi = sine_fit(read_data)

        # Handle both single column and multi-column results
        if read_data.shape[1] == 1 or np.isscalar(freq):
            # Single column case
            freq_arr = np.array([freq])
            mag_arr = np.array([mag])
            dc_arr = np.array([dc])
            phi_arr = np.array([phi])
            data_fit_2d = data_fit.reshape(-1, 1)
        else:
            # Multi-column case
            freq_arr = freq
            mag_arr = mag
            dc_arr = dc
            phi_arr = phi
            data_fit_2d = data_fit

        # Save each variable to separate CSV (matching MATLAB format)
        save_variable(sub_folder, freq_arr, 'freq')
        save_variable(sub_folder, mag_arr, 'mag')
        save_variable(sub_folder, dc_arr, 'dc')
        save_variable(sub_folder, phi_arr, 'phi')
        save_variable(sub_folder, data_fit_2d, 'data_fit')

        # Print metrics summary
        if len(freq_arr) == 1:
            print(f"  freq={freq_arr[0]:.8f}, mag={mag_arr[0]:.6f}, "
                  f"dc={dc_arr[0]:.6f}, phi={phi_arr[0]:.6f} rad\n")
        else:
            print(f"  {len(freq_arr)} columns fitted:")
            print(f"  freq: mean={np.mean(freq_arr):.8f}, std={np.std(freq_arr):.8f}")
            print(f"  mag:  mean={np.mean(mag_arr):.6f}, std={np.std(mag_arr):.6f}")
            print(f"  dc:   mean={np.mean(dc_arr):.6f}, std={np.std(dc_arr):.6f}")
            print(f"  phi:  mean={np.mean(phi_arr):.6f}, std={np.std(phi_arr):.6f}\n")

    print("test_sineFit complete.")


if __name__ == '__main__':
    test_sineFit()
