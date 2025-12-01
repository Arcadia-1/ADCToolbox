"""test_spec_plot.py - Unit test for spec_plot function

Tests the spec_plot function with various sinewave datasets.

Output structure:
  test_output/test_spec_plot/<dataset_name>/
      ENoB_python.csv
      SNDR_python.csv
      SFDR_python.csv
      SNR_python.csv
      THD_python.csv
      pwr_python.csv
      NF_python.csv
      spectrum_python.png
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from adctoolbox.aout import spec_plot
from tests._utils import auto_search_files, save_variable, save_fig

plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True


def test_spec_plot(project_root):
    """Test spec_plot function on sinewave datasets."""
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

            # Read data as 2D array (spec_plot expects 2D)
            read_data = np.loadtxt(data_file_path, delimiter=',', ndmin=2)

            dataset_name = data_file_path.stem
            sub_folder = output_dir / dataset_name / "test_spec_plot"
            sub_folder.mkdir(parents=True, exist_ok=True)

            # Run spec_plot
            plt.figure(figsize=(12, 8))
            ENoB, SNDR, SFDR, SNR, THD, pwr, NF, _ = spec_plot(
                read_data,
                label=1,
                harmonic=5,
                OSR=1,
                NFMethod=0
            )

            plt.title(f'Spectrum: {dataset_name}')

            # Save outputs
            save_fig(sub_folder, 'spectrum_python.png', dpi=100)
            save_variable(sub_folder, ENoB, 'ENoB')
            save_variable(sub_folder, SNDR, 'SNDR')
            save_variable(sub_folder, SFDR, 'SFDR')
            save_variable(sub_folder, SNR, 'SNR')
            save_variable(sub_folder, THD, 'THD')
            save_variable(sub_folder, pwr, 'pwr')
            save_variable(sub_folder, NF, 'NF')

            success_count += 1

        except Exception as e:
            print(f"  [ERROR] {str(e)}")
            continue

    print("-" * 60)
    print(f"[DONE] Test complete. Success: {success_count}/{len(files_list)}")
    plt.close('all')

