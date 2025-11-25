"""Test INLSine.py - Python version of test_INL_from_sine.m"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from adctoolbox.aout import INLsine


def test_inl_from_sine():
    """Test INLSine function with sinewave data files."""

    # Configuration - assumes running from project root d:\ADCToolbox
    inputdir = Path("test_data")
    outputdir = Path("test_output")

    # Resolution list to scan
    Resolution_list = [12]

    # Create output directory if it doesn't exist
    outputdir.mkdir(exist_ok=True)

    # Manual file list (empty means use search patterns)
    manual_files_list = []

    if manual_files_list:
        file_list = manual_files_list
    else:
        # Search patterns for input files
        search_patterns = ["sinewave_HD_*.csv", "sinewave_gain_error_*.csv"]
        all_files = []

        for pattern in search_patterns:
            matched_files = list(inputdir.glob(pattern))
            all_files.extend([f.name for f in matched_files])

        # Remove duplicates while preserving order
        seen = set()
        file_list = []
        for f in all_files:
            if f not in seen:
                seen.add(f)
                file_list.append(f)

    # Process each file
    for current_filename in file_list:
        data_filepath = inputdir / current_filename

        # Extract name without extension
        name = Path(current_filename).stem

        # Load data
        data = np.loadtxt(data_filepath, delimiter=',')
        if data.ndim > 1:
            data = data.flatten()

        # Process each resolution
        for Resolution in Resolution_list:
            # Scale data by 2^Resolution
            scaled_data = data * (2 ** Resolution)

            # Calculate min/max for verification
            data_min = np.min(scaled_data)
            data_max = np.max(scaled_data)
            expected_max = 2 ** Resolution

            # Call INLsine function
            INL, DNL, code = INLsine(scaled_data)

            # Calculate DNL/INL ranges
            max_inl = np.max(INL)
            min_inl = np.min(INL)
            max_dnl = np.max(DNL)
            min_dnl = np.min(DNL)

            # Create figure with 2 subplots
            fig = plt.figure(figsize=(10, 8))

            # Top subplot: INL
            plt.subplot(2, 1, 1)
            plt.scatter(code, INL, s=8, alpha=0.6)
            plt.xlabel('Code')
            plt.ylabel('INL (LSB)')
            plt.grid(True)
            title_str_inl = f'INL = [{min_inl:.2f}, {max_inl:+.2f}] LSB'
            plt.title(title_str_inl)

            # Set y-axis limits (at least [-1, 1])
            ylim_min = min(min_inl, -1)
            ylim_max = max(max_inl, 1)
            plt.ylim([ylim_min, ylim_max])
            plt.xlim([0, expected_max])

            # Bottom subplot: DNL
            plt.subplot(2, 1, 2)
            plt.scatter(code, DNL, s=8, alpha=0.6)
            plt.xlabel('Code')
            plt.ylabel('DNL (LSB)')
            plt.grid(True)
            title_str_dnl = f'DNL = [{min_dnl:.2f}, {max_dnl:.2f}] LSB'
            plt.title(title_str_dnl)

            # Set y-axis limits (at least [-1, 1])
            ylim_min = min(min_dnl, -1)
            ylim_max = max(max_dnl, 1)
            plt.ylim([ylim_min, ylim_max])
            plt.xlim([0, expected_max])

            # Save figure
            subdir_path = outputdir / name
            subdir_path.mkdir(parents=True, exist_ok=True)

            output_filename_base = f'INL_{Resolution}b_{name}_python'
            output_filepath = subdir_path / f'{output_filename_base}.png'

            plt.tight_layout()
            plt.savefig(output_filepath, dpi=150)
            plt.close(fig)

            # Console output
            print(f'[Resolution = {Resolution}]: DNL = [{min_dnl:.2f}, {max_dnl:.2f}] LSB, INL = [{min_inl:.2f}, {max_inl:+.2f}] LSB')
            print(f'[Saved image] -> [{output_filepath}]\n')

    # Close any remaining figures
    plt.close('all')


if __name__ == "__main__":
    test_inl_from_sine()
