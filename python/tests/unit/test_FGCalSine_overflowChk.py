"""test_FGCalSine_overflowChk.py - Unit test for overflowChk function

Tests the overflowChk function with SAR ADC digital output data.

Output structure:
    test_output/<data_set_name>/test_overflowChk/
        overflowChk_python.png  - overflow check plot

Configuration - assumes running from project root d:\ADCToolbox
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
import sys
from pathlib import Path
from glob import glob

from adctoolbox.dout import FGCalSine, overflowChk


def test_fgcalsine_overflowchk():
    """Test FGCalSine and overflowChk with SAR ADC data."""

    print('=== test_FGCalSine_overflowChk.py ===')

    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')

    # Configuration
    input_dir = Path("test_data")
    output_dir = Path("test_output")

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
            sub_folder = output_dir / dataset_name / 'test_overflowChk'
            sub_folder.mkdir(parents=True, exist_ok=True)

            # Run FGCalSine to get calibrated weights
            weights_cal = FGCalSine(read_data)[0]  # Only need weights

            # Run overflowChk
            fig = plt.figure(figsize=(10, 6))
            plt.ioff()  # Turn off interactive mode
            overflowChk(read_data, weights_cal)
            plt.title(title_string)

            # Save plot
            plot_path = sub_folder / 'overflowChk_python.png'
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

    print(f'[test_FGCalSine_overflowChk COMPLETE] {success_count}/{len(files_list)} passed')
    return success_count == len(files_list)


if __name__ == "__main__":
    sys.exit(0 if test_fgcalsine_overflowchk() else 1)
