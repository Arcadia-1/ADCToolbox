"""test_fg_cal_sine_overflow_chk.py - Unit test for overflow_chk function

Tests the overflow_chk function with SAR ADC digital output data.

Output structure:
    test_output/<data_set_name>/test_overflow_chk/
        overflow_chk_python.png  - overflow check plot

Configuration - assumes running from project root d:\ADCToolbox
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
import sys
from pathlib import Path
from glob import glob

from adctoolbox.dout import fg_cal_sine, overflow_chk


# Get project root directory (two levels up from python/tests/unit)
project_root = Path(__file__).resolve().parents[3]

def test_fgcalsine_overflowchk():
    """Test FGCalSine and overflow_chk with SAR ADC data."""

    print('=== test_fg_cal_sine_overflow_chk.py ===')

    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')

    # Configuration
    input_dir = project_root / "dataset"
    output_dir = project_root / "test_output"

    # Auto-search for dout_SAR_*.csv files
    file_pattern = str(input_dir / "dout_SAR_*.csv")
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
            title_string = dataset_name  # Keep underscores as-is in Python

            # Create output subfolder
            sub_folder = output_dir / dataset_name / 'test_FGCalSine_overflowChk'
            sub_folder.mkdir(parents=True, exist_ok=True)

            # Run FGCalSine to get calibrated weights
            weights_cal = fg_cal_sine(read_data)[0]  # Only need weights

            # Run overflow_chk
            fig = plt.figure(figsize=(10, 6))
            plt.ioff()  # Turn off interactive mode
            overflow_chk(read_data, weights_cal)
            plt.title(title_string)

            # Save plot
            plot_path = sub_folder / 'overflow_chk_python.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f'  [Saved] {plot_path}\n')
            plt.close(fig)

            success_count += 1

        except Exception as e:
            print(f'  [ERROR] {e}\n')
            import traceback
            traceback.print_exc()
            plt.close('all')
            continue

    # Re-enable warnings
    warnings.filterwarnings('default')

    # Close any remaining figures
    plt.close('all')

    print(f'[test_FGCalSine_overflow_chk COMPLETE] {success_count}/{len(files_list)} passed')
    return success_count == len(files_list)


if __name__ == "__main__":
    sys.exit(0 if test_fgcalsine_overflowchk() else 1)
