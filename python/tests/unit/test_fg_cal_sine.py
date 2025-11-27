"""test_fg_cal_sine.py - Unit test for FGCalSine function

Tests the FGCalSine foreground calibration function with SAR/Pipeline digital code data.

Output structure:
    test_output/<data_set_name>/test_FGCalSine/
        weight_python.csv       - calibrated bit weights
        offset_python.csv       - DC offset
        freqCal_python.csv      - calibrated frequency
        postCal_python.csv      - first 1000 samples of calibrated output
        ideal_python.csv        - first 1000 samples of ideal sinewave
        err_python.csv          - first 1000 samples of residual error

Configuration - assumes running from project root d:\ADCToolbox
"""

import numpy as np
import sys
from pathlib import Path
from glob import glob

from adctoolbox.dout import fg_cal_sine
from save_variable import save_variable

# Get project root directory (two levels up from python/tests/unit)
project_root = Path(__file__).resolve().parents[3]

def run_fgcal_tests():
    """Test FGCalSine function on digital code datasets."""

    print('=== test_fg_cal_sine.py ===')

    # Configuration
    input_dir = project_root / "dataset"
    output_dir = project_root / "test_output"

    # Auto-search for dout_*.csv files
    file_pattern = str(input_dir / "dout_*.csv")
    files_list = sorted(glob(file_pattern))

    if not files_list:
        print(f"[ERROR] No files found matching pattern: {file_pattern}")
        return False

    print(f'[Testing] {len(files_list)} datasets...\n')

    success_count = 0

    for k, data_file_path in enumerate(files_list, 1):
        current_filename = Path(data_file_path).name

        if not Path(data_file_path).exists():
            print(f'[{k}/{len(files_list)}] {current_filename} - NOT FOUND, skipping\n')
            continue

        print(f'[{k}/{len(files_list)}] [Processing] {current_filename}')

        try:
            # Read data
            read_data = np.loadtxt(data_file_path, delimiter=',')

            # Extract dataset name
            dataset_name = Path(current_filename).stem

            # Create output subfolder
            sub_folder = output_dir / dataset_name / 'test_FGCalSine'
            sub_folder.mkdir(parents=True, exist_ok=True)

            # Run FGCalSine
            weight, offset, postCal, ideal, err, freqCal = fg_cal_sine(
                read_data,
                freq=0,
                order=5
            )

            # Save each variable to separate CSV (matching MATLAB format)
            save_variable(sub_folder, weight, 'weight')
            save_variable(sub_folder, offset, 'offset')
            save_variable(sub_folder, freqCal, 'freqCal')
            save_variable(sub_folder, postCal, 'postCal')
            save_variable(sub_folder, ideal, 'ideal')
            save_variable(sub_folder, err, 'err')

            # Print summary
            err_rms = np.sqrt(np.mean(err**2))
            print(f'  [Results] freqCal={freqCal:.8f}, offset={offset:.6f}, '
                  f'weight_sum={np.sum(weight):.6f}, err_rms={err_rms:.6f}\n')

            success_count += 1

        except Exception as e:
            print(f'  [ERROR] {e}\n')
            import traceback
            traceback.print_exc()
            continue

    print(f'[test_FGCalSine COMPLETE] {success_count}/{len(files_list)} passed')
    return success_count == len(files_list)


if __name__ == "__main__":
    sys.exit(0 if run_fgcal_tests() else 1)
