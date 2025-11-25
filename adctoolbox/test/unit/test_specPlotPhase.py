"""test_specPlotPhase.py - Unit test for spec_plot_phase function

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
project_root = Path(__file__).parent.parent.parent.parent


def main():
    """Main test function."""
    input_dir = project_root / "test_data"
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
    print("=== test_specPlotPhase.py ===")
    print(f"Testing spec_plot_phase function with {len(files_list)} datasets...\n")

    for k, current_filename in enumerate(files_list, start=1):
        data_file_path = input_dir / current_filename

        if not data_file_path.is_file():
            print(f"[{k}/{len(files_list)}] {current_filename} - NOT FOUND\n")
            continue

        print(f"[{k}/{len(files_list)}] {current_filename}")

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

        # Save phase data to CSV
        phase_data_path = sub_folder / 'phase_data_python.csv'
        phase_df.to_csv(phase_data_path, index=False)

        print(f"  [Saved] {phase_plot_path}")
        print(f"  [Saved] {phase_data_path} (fundamental bin: {result['bin']})")

        # Print harmonic information
        harm_str = " ".join([f"H{h['harmonic']}={h['magnitude']:.1f}dB" for h in result['harmonics']])
        print(f"  Harmonics: {harm_str}")

        # Compare with MATLAB results if available
        matlab_csv_path = sub_folder / 'phase_data_matlab.csv'
        if matlab_csv_path.exists():
            try:
                matlab_df = pd.read_csv(matlab_csv_path)
                python_df = pd.read_csv(phase_data_path)

                # Skip first row (DC bin) which may have NaN in MATLAB
                matlab_df = matlab_df.iloc[1:]
                python_df = python_df.iloc[1:]

                if len(matlab_df) == len(python_df):
                    mag_diff = np.sqrt(np.mean((matlab_df['spec_mag'].values - python_df['spec_mag'].values)**2))
                    phase_diff = np.sqrt(np.mean((matlab_df['spec_phase'].values - python_df['spec_phase'].values)**2))

                    comparison = pd.DataFrame({
                        'Metric': ['RMS_Magnitude_Diff', 'RMS_Phase_Diff_rad', 'Length'],
                        'Value': [mag_diff, phase_diff, len(python_df)]
                    })

                    comparison_path = sub_folder / 'comparison.csv'
                    comparison.to_csv(comparison_path, index=False)
                    print(f"    [Saved] {comparison_path}")

                    # Determine status
                    if mag_diff < 0.01 and phase_diff < 0.01:
                        status = "EXCELLENT"
                    elif mag_diff < 0.1 and phase_diff < 0.1:
                        status = "GOOD"
                    else:
                        status = "NEEDS REVIEW"

                    print(f"    [Comparison] RMS mag diff: {mag_diff:.4f}, RMS phase diff: {phase_diff:.4f} rad - {status}")
                else:
                    print(f"    [Warning] Length mismatch: MATLAB={len(matlab_df)}, Python={len(python_df)}")
            except Exception as e:
                print(f"    [Warning] Comparison failed: {e}")
        else:
            print(f"    [Warning] No MATLAB results found for comparison")

        print()

    print("test_specPlotPhase.py complete.")

    # Close any remaining figures
    plt.close('all')


if __name__ == "__main__":
    main()
