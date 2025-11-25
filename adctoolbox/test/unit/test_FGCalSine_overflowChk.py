"""Test FGCalSine and overflowChk - Python version of test_FGCalSine_overflowChk.m"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

from adctoolbox.dout import FGCalSine, overflowChk


def test_fgcalsine_overflowchk():
    """Test FGCalSine and overflowChk with SAR ADC data."""

    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')

    # Define input and output directories
    inputdir = Path("test_data")
    outputdir = Path("test_output")

    # File list
    files_list = [
        'dout_SAR_12b_weight_1.csv',
        'dout_SAR_12b_weight_2.csv',
        'dout_SAR_12b_weight_3.csv',
    ]

    # If files_list is empty, search for dout*.csv files
    if not files_list:
        search_pattern = "dout*.csv"
        matched_files = list(inputdir.glob(search_pattern))
        files_list = [f.name for f in matched_files]

    # Process each file
    for current_filename in files_list:
        data_filepath = inputdir / current_filename

        print(f"[Processing] {current_filename}")

        # Load data
        read_code = np.loadtxt(data_filepath, delimiter=',')
        if read_code.ndim == 1:
            read_code = read_code.reshape(-1, 1)

        print(f"  [Data shape] {read_code.shape}")

        # Call FGCalSine to get calibrated weights
        print(f"  [FGCalSine] Starting calibration...")
        weights_cal, offset, postCal, ideal, err, freqCal = FGCalSine(read_code)

        print(f"  [FGCalSine] Calibrated weights: {weights_cal}")
        print(f"  [FGCalSine] Offset: {offset:.6f}")
        print(f"  [FGCalSine] Calibrated frequency: {freqCal:.6f}")

        # Create figure for overflowChk
        fig = plt.figure(figsize=(10, 6))

        # Call overflowChk to visualize residue distribution
        overflowChk(read_code, weights_cal)

        # Set title (matching MATLAB format with underscores)
        name = current_filename.replace('.csv', '')
        title_string = name  # Keep underscores as-is (no escaping needed in Python)
        plt.title(title_string)

        # Save figure
        output_filepath = outputdir / f'{name}_overflowChk_python.png'
        plt.savefig(output_filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"  [Saved image] -> [{output_filepath}]\n")

    # Re-enable warnings
    warnings.filterwarnings('default')

    # Close any remaining figures
    plt.close('all')


if __name__ == "__main__":
    test_fgcalsine_overflowchk()
