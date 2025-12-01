"""test_spec_plot_phase.py - Unit test for spec_plot_phase function

Tests the spec_plot_phase function with various sinewave datasets.

Output structure:
  test_output/test_spec_plot_phase/<dataset_name>/
      phase_python.png
      freq_bin_python.csv
      spec_real_python.csv
      spec_imag_python.csv
      spec_mag_python.csv
      spec_phase_python.csv
      phi_real_python.csv
      phi_imag_python.csv
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from adctoolbox.aout import spec_plot_phase
from tests._utils import auto_search_files, save_variable

plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True


def test_spec_plot_phase(project_root):
    """Test spec_plot_phase function on sinewave datasets."""
    input_dir = project_root / "dataset" / "aout" / "sinewave"
    output_dir = project_root / "test_output"

    files_list = []
    files_list = auto_search_files(files_list, input_dir, 'sinewave_*.csv')

    output_dir.mkdir(parents=True, exist_ok=True)
    success_count = 0

    for k, current_filename in enumerate(files_list, 1):
        try:
            data_file_path = input_dir / current_filename
            print(f"[{k}/{len(files_list)}] Processing: [{current_filename}]")

            # Read data as 2D array
            read_data = np.loadtxt(data_file_path, delimiter=',', ndmin=2)

            dataset_name = data_file_path.stem
            sub_folder = output_dir / dataset_name / "test_spec_plot_phase"
            sub_folder.mkdir(parents=True, exist_ok=True)

            # Run spec_plot_phase
            phase_plot_path = sub_folder / 'phase_python.png'
            result = spec_plot_phase(read_data, save_path=str(phase_plot_path))

            # Extract data for CSV output
            spec = result['spec']
            freq_bins = result['freq_bins']

            # Remove first element (DC bin) to match MATLAB format
            freq_bins_no_dc = freq_bins[1:]
            spec_no_dc = spec[1:len(freq_bins)]
            phi_no_dc = result['phi'][1:len(freq_bins)]

            # Save each variable
            save_variable(sub_folder, freq_bins_no_dc, 'freq_bin')
            save_variable(sub_folder, np.real(spec_no_dc), 'spec_real')
            save_variable(sub_folder, np.imag(spec_no_dc), 'spec_imag')
            save_variable(sub_folder, np.abs(spec_no_dc), 'spec_mag')
            save_variable(sub_folder, np.angle(spec_no_dc), 'spec_phase')
            save_variable(sub_folder, np.real(phi_no_dc), 'phi_real')
            save_variable(sub_folder, np.imag(phi_no_dc), 'phi_imag')

            success_count += 1

        except Exception as e:
            print(f"  [ERROR] {str(e)}")
            continue

    print("-" * 60)
    print(f"[DONE] Test complete. Success: {success_count}/{len(files_list)}")
    plt.close('all')

