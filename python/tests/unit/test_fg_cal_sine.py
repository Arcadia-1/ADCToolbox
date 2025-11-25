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
import pandas as pd
import sys
from pathlib import Path
from glob import glob

from adctoolbox.dout import fg_cal_sine

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

            # Save weight to CSV
            weight_table = pd.DataFrame({
                'bit_index': np.arange(1, len(weight) + 1),
                'weight': weight
            })
            weight_path = sub_folder / 'weight_python.csv'
            weight_table.to_csv(weight_path, index=False)
            print(f'  [Saved] {weight_path}')

            # Save offset to CSV
            offset_table = pd.DataFrame({'offset': [offset]})
            offset_path = sub_folder / 'offset_python.csv'
            offset_table.to_csv(offset_path, index=False)
            print(f'  [Saved] {offset_path}')

            # Save freqCal to CSV
            freqCal_table = pd.DataFrame({'freqCal': [freqCal]})
            freqCal_path = sub_folder / 'freqCal_python.csv'
            freqCal_table.to_csv(freqCal_path, index=False)
            print(f'  [Saved] {freqCal_path}')

            # Save postCal, ideal, err (first 1000 samples)
            N_save = min(1000, len(postCal))

            postCal_table = pd.DataFrame({'postCal': postCal[:N_save]})
            postCal_path = sub_folder / 'postCal_python.csv'
            postCal_table.to_csv(postCal_path, index=False)
            print(f'  [Saved] {postCal_path}')

            ideal_table = pd.DataFrame({'ideal': ideal[:N_save]})
            ideal_path = sub_folder / 'ideal_python.csv'
            ideal_table.to_csv(ideal_path, index=False)
            print(f'  [Saved] {ideal_path}')

            err_table = pd.DataFrame({'err': err[:N_save]})
            err_path = sub_folder / 'err_python.csv'
            err_table.to_csv(err_path, index=False)
            print(f'  [Saved] {err_path}')

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
