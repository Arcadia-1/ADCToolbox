"""test_spec_plot_phase.py - Unit test for spec_plot_phase function

Tests the spec_plot_phase function with various sinewave datasets.

Output structure:
  test_output/<data_set_name>/test_specPlotPhase/
      phase_python.png        - Phase polar plot
      phase_data_python.csv   - Spectrum data with phase information
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from adctoolbox.aout import spec_plot_phase

# Get project root directory
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
            print(f"[{k}/{len(files_list)}] [test_specPlotPhase] [ERROR] File not found: {current_filename}")
            continue

        # Read data
        read_data = np.loadtxt(data_file_path, delimiter=',', ndmin=2)

        # Extract dataset name
        dataset_name = data_file_path.stem

        # Create output subfolder
        sub_folder = output_dir / dataset_name / 'test_specPlotPhase'
        sub_folder.mkdir(parents=True, exist_ok=True)

        # Run spec_plot_phase
        phase_plot_path = sub_folder / 'phase_python.png'
        result = spec_plot_phase(read_data, save_path=str(phase_plot_path))

        # Extract data for CSV output
        spec = result['spec']
        freq_bins = result['freq_bins']

        # Create dataframe matching MATLAB output format
        phase_df = pd.DataFrame({
            'freq_bin': freq_bins,
            'spec_real': np.real(spec[:len(freq_bins)]),
            'spec_imag': np.imag(spec[:len(freq_bins)]),
            'spec_mag': np.abs(spec[:len(freq_bins)]),
            'spec_phase': np.angle(spec[:len(freq_bins)]),
            'phi_real': np.real(result['phi'][:len(freq_bins)]),
            'phi_imag': np.imag(result['phi'][:len(freq_bins)])
        })

        # Remove first row (DC bin) to match MATLAB format
        phase_df = phase_df.iloc[1:]

        # Save phase data to CSV
        phase_data_path = sub_folder / 'phase_data_python.csv'
        phase_df.to_csv(phase_data_path, index=False)

        # Get fundamental frequency magnitude for display
        fund_mag_dB = result['harmonics'][0]['magnitude'] if result['harmonics'] else 0.0

        # Print one-line progress
        print(f"[{k}/{len(files_list)}] [test_specPlotPhase] [Fund={fund_mag_dB:.1f}dB] from {current_filename}")

    print("test_spec_plot_phase.py complete.")

    # Close any remaining figures
    plt.close('all')


if __name__ == "__main__":
    main()
