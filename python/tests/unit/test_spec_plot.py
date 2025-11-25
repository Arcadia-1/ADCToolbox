"""test_spec_plot.py - Unit test for spec_plot function

Tests the spec_plot function with various sinewave datasets.

Output structure:
  test_output/<data_set_name>/test_specPlot/
      metrics_python.csv      - ENoB, SNDR, SFDR, SNR, THD, pwr, NF
      spectrum_python.png     - Spectrum plot
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from adctoolbox.aout import spec_plot

# Get project root directory (two levels up from python/tests/unit)
project_root = Path(__file__).resolve().parents[3]


def main():
    """Main test function."""
    input_dir = project_root / "dataset"
    output_dir = project_root / "test_output"

    # Test datasets - leave empty to auto-search
    files_list = []

    # Auto-search if list is empty
    if not files_list:
        search_patterns = ['sinewave_*.csv', 'batch_sinewave_*.csv']
        files_list = []
        for pattern in search_patterns:
            files_list.extend([f.name for f in input_dir.glob(pattern)])
        print(f"Auto-discovered {len(files_list)} files\n")

    if not files_list:
        print(f"ERROR: No test files found in {input_dir}")
        return

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Test Loop
    for k, current_filename in enumerate(files_list, start=1):
        data_file_path = input_dir / current_filename

        if not data_file_path.is_file():
            print(f"[{k}/{len(files_list)}] [test_specPlot] [ERROR] File not found: {current_filename}")
            continue

        # Read data
        read_data = np.loadtxt(data_file_path, delimiter=',', ndmin=2)

        # Extract dataset name
        dataset_name = data_file_path.stem

        # Create output subfolder
        sub_folder = output_dir / dataset_name / 'test_specPlot'
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

        # Update title
        plt.title(f'specPlot: {dataset_name}')

        # Save plot
        plot_path = sub_folder / 'spectrum_python.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()

        # Save metrics to CSV
        metrics_df = pd.DataFrame([{
            'ENoB': ENoB, 'SNDR': SNDR, 'SFDR': SFDR,
            'SNR': SNR, 'THD': THD, 'pwr': pwr, 'NF': NF
        }])
        metrics_path = sub_folder / 'metrics_python.csv'
        metrics_df.to_csv(metrics_path, index=False)

        # Print one-line progress
        print(f"[{k}/{len(files_list)}] [test_specPlot] [ENoB={ENoB:.2f}] from {current_filename}")

    print("[test_specPlot complete]")

    # Close any remaining figures
    plt.close('all')


if __name__ == "__main__":
    main()
