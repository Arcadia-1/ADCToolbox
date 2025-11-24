"""
Test batch data processing - Python version of test_batch_data.m

Tests specPlot and specPlotPhase with batch/multi-run data.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add project root to sys.path if needed (for direct script execution)
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from ADCToolbox_Python.spec_plot import spec_plot
from ADCToolbox_Python.specPlotPhase import spec_plot_phase


def main():
    """Test batch data processing."""

    # Configuration
    input_dir = os.path.join(_project_root, "ADCToolbox_example_data")
    output_dir = os.path.join(_project_root, "ADCToolbox_example_output")

    # Define files to process
    files_list = [
        'batch_sinewave_Nrun_2.csv',
        'batch_sinewave_Nrun_16.csv',
        'batch_sinewave_Nrun_100.csv'
    ]

    # Check if output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Processing loop
    print(f"Starting batch processing ({len(files_list)} files)...")
    for current_filename in files_list:
        print(f"Processing file: {current_filename}")

        # Load data
        data_file_path = os.path.join(input_dir, current_filename)
        read_data = np.loadtxt(data_file_path, delimiter=',')

        # Extract file name for subfolder
        name = os.path.splitext(current_filename)[0]
        title_string = name.replace('_', '\_')  # Escape underscore for plot title

        # Create subfolder
        sub_folder = os.path.join(output_dir, name)
        os.makedirs(sub_folder, exist_ok=True)

        # ----------------------------------------------------------------
        # PLOT 1: SPECTRUM
        # ----------------------------------------------------------------
        plt.figure(figsize=(12, 8))

        # Standard Spectrum Plot
        spec_plot(read_data, label=1, harmonic=0, OSR=1, coAvg=0)
        plt.title(f'Spectrum Plot: {title_string}', fontsize=12)

        # Save Plot 1
        output_filename1 = f'Spectrum_of_{name}_python.png'
        output_filepath1 = os.path.join(sub_folder, output_filename1)
        plt.savefig(output_filepath1, dpi=150)
        print(f'[Saved image 1] -> [{output_filepath1}]')
        plt.close()

        # ----------------------------------------------------------------
        # PLOT 2: PHASE
        # ----------------------------------------------------------------
        plt.figure(figsize=(8, 8))

        # Phase Plot
        spec_plot_phase(read_data, harmonic=50, show_plot=False)
        plt.title(f'Phase Plot: {title_string}', fontsize=12)

        # Save Plot 2
        output_filename2 = f'Phase_of_{name}_python.png'
        output_filepath2 = os.path.join(sub_folder, output_filename2)
        plt.savefig(output_filepath2, dpi=150)
        print(f'[Saved image 2] -> [{output_filepath2}]\n')
        plt.close()

    print("Batch processing complete.")


if __name__ == "__main__":
    main()
