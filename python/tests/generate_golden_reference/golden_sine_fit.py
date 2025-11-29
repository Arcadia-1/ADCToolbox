"""golden_sine_fit.py - Golden reference test for sineFit

This test processes only the datasets listed in golden_data_list.txt
Use this to generate golden references, not for comprehensive testing.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Get project root directory (three levels up from python/tests/generate_golden_reference)
project_root = Path(__file__).resolve().parents[3]

# Add unit tests to path for save_variable
unit_tests_dir = project_root / 'python' / 'tests' / 'unit'
sys.path.insert(0, str(unit_tests_dir))

from adctoolbox.common import sine_fit
from save_variable import save_variable


def golden_sineFit():
    """Test sineFit on golden reference datasets only."""

    # Configuration
    input_dir = project_root / "dataset"
    output_dir = project_root / "test_reference"

    # Read golden data list
    golden_list_file = project_root / "test_reference" / "golden_data_list.txt"
    if not golden_list_file.exists():
        raise FileNotFoundError(f"Golden data list not found: {golden_list_file}")

    files_list = []
    with open(golden_list_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                files_list.append(line)

    print(f"Golden test: processing {len(files_list)} files from golden_data_list.txt\n")

    if not files_list:
        raise ValueError(f"No files found in golden_data_list.txt")

    output_dir.mkdir(exist_ok=True)

    # Test Loop
    print('=== golden_sine_fit.py ===')
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

        # Create output subfolder (match unit test structure)
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

        # Generate plot (matching MATLAB visualization)
        if read_data.shape[1] == 1:
            fig, ax = plt.subplots(figsize=(10, 7.5))

            period = round(1 / freq)
            n_samples = min(max(period, 20), len(read_data))
            t = np.arange(1, n_samples + 1)

            # Plot original data
            ax.plot(t, read_data[:n_samples, 0], '-o', linewidth=2, label='Original Data')

            # Plot fitted sine
            t_dense = np.linspace(1, n_samples, n_samples * 100)
            fitted_sine = mag * np.cos(2 * np.pi * freq * (t_dense - 1) + phi) + dc
            ax.plot(t_dense, fitted_sine, '--', linewidth=1, label='Fitted Sine')

            ax.set_xlabel('Sample', fontsize=16)
            ax.set_ylabel('Amplitude', fontsize=16)
            ax.legend(loc='upper left', fontsize=14)
            ax.grid(True)
            ax.set_ylim([fitted_sine.min() - 0.1, fitted_sine.max() + 0.2])
            ax.tick_params(labelsize=14)

            # Save plot
            plot_path = sub_folder / 'sineFit_python.png'
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

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

    print("golden_sineFit complete.")


if __name__ == '__main__':
    golden_sineFit()
