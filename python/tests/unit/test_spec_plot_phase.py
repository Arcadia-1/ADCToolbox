"""test_spec_plot_phase.py - Unit test for spec_plot_phase function

Tests the spec_plot_phase function with various sinewave datasets.

Output structure:
  test_output/<data_set_name>/test_specPlotPhase/
      phase_python.png        - Phase polar plot
      freq_bin_python.csv     - Frequency bins
      spec_real_python.csv    - Spectrum real part
      spec_imag_python.csv    - Spectrum imaginary part
      spec_mag_python.csv     - Spectrum magnitude
      spec_phase_python.csv   - Spectrum phase
      phi_real_python.csv     - Phi real part
      phi_imag_python.csv     - Phi imaginary part
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from adctoolbox.aout import spec_plot_phase
from save_variable import save_variable

# Get project root directory
project_root = Path(__file__).resolve().parents[3]


def main():
    """Main test function."""
    input_dir = project_root / "dataset" / "aout"
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

        # Remove first element (DC bin) to match MATLAB format
        freq_bins_no_dc = freq_bins[1:]
        spec_no_dc = spec[1:len(freq_bins)]
        phi_no_dc = result['phi'][1:len(freq_bins)]

        # Save each variable to separate CSV (matching MATLAB format)
        save_variable(sub_folder, freq_bins_no_dc, 'freq_bin')
        save_variable(sub_folder, np.real(spec_no_dc), 'spec_real')
        save_variable(sub_folder, np.imag(spec_no_dc), 'spec_imag')
        save_variable(sub_folder, np.abs(spec_no_dc), 'spec_mag')
        save_variable(sub_folder, np.angle(spec_no_dc), 'spec_phase')
        save_variable(sub_folder, np.real(phi_no_dc), 'phi_real')
        save_variable(sub_folder, np.imag(phi_no_dc), 'phi_imag')

        # Get fundamental frequency magnitude for display
        fund_mag_dB = result['harmonics'][0]['magnitude'] if result['harmonics'] else 0.0

        # Print one-line progress
        print(f"[{k}/{len(files_list)}] [test_specPlotPhase] [Fund={fund_mag_dB:.1f}dB] from {current_filename}")

    print("test_spec_plot_phase.py complete.")

    # Close any remaining figures
    plt.close('all')


if __name__ == "__main__":
    main()
