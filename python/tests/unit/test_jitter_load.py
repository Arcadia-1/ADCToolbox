"""test_jitter_load.py - Test jitter analysis with deterministic data

Input: test_data/jitter_sweep/
Output: test_output/jitter_sweep/
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

from adctoolbox.common import sine_fit
from adctoolbox.aout import spec_plot, err_hist_sine
from tests._utils import save_variable, save_fig

# Get project root directory
project_root = Path(__file__).resolve().parents[3]

def main():
    """Main test function."""
    input_dir = project_root / "dataset" / "jitter_sweep"
    output_dir = project_root / "test_output" / "jitter_sweep"

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # Load configuration
    config_filepath = input_dir / 'config.csv'
    if not config_filepath.exists():
        print('[ERROR] Config file not found. Please run generate_jitter_sweep_data first')
        return

    config_data = pd.read_csv(config_filepath)
    Fs_expected = config_data.loc[config_data.iloc[:, 0] == 'Fs', config_data.columns[1]].values[0]
    N_expected = int(config_data.loc[config_data.iloc[:, 0] == 'N', config_data.columns[1]].values[0])

    # Load metadata
    metadata_filepath = input_dir / 'jitter_sweep_metadata.csv'
    metadata = pd.read_csv(metadata_filepath)  # Read with header
    Tj_list = metadata['Tj_seconds'].values  # Use column name

    # Load frequency list
    freq_metadata_filepath = input_dir / 'frequency_list.csv'
    freq_metadata = pd.read_csv(freq_metadata_filepath)
    Fin_list_nominal = freq_metadata.iloc[:, 0].astype(float).values

    print('=== test_jitter_load.py ===')
    print(f'[Input dir] {input_dir}')
    print(f'[Output dir] {output_dir}')
    print(f'[Fs from config] {Fs_expected:.3e} Hz')
    print(f'[N from config] {N_expected}\n')

    # Analyze each frequency
    for i_freq, Fin_nominal in enumerate(Fin_list_nominal):
        print(f'[Analyzing] Nominal Fin = {round(Fin_nominal/1e6)} MHz')

        meas_jitter = np.zeros(len(Tj_list))
        meas_SNDR = np.zeros(len(Tj_list))
        set_jitter = np.zeros(len(Tj_list))
        actual_Fin = np.zeros(len(Tj_list))
        pnoi_array = np.zeros(len(Tj_list))
        anoi_array = np.zeros(len(Tj_list))

        for i_tj, Tj in enumerate(Tj_list):
            set_jitter[i_tj] = Tj

            filename = f'jitter_sweep_Fin_{round(Fin_nominal/1e6)}MHz_Tj_idx_{i_tj+1:02d}.csv'
            filepath = input_dir / filename

            if not filepath.exists():
                print(f'[WARNING] File not found: {filepath}')
                continue

            read_data = np.loadtxt(filepath, delimiter=',')
            N = len(read_data)

            if N != N_expected:
                print(f'[WARNING] N mismatch: Data has N={N}, config expects N={N_expected}')

            # Sine fit
            data_fit, f_norm, mag, dc, phi = sine_fit(read_data)
            Fin_fit = f_norm * Fs_expected
            actual_Fin[i_tj] = Fin_fit

            # Error histogram analysis
            emean, erms, phase_code, anoi, pnoi, err, xx = err_hist_sine(
                read_data, bin=99, fin=f_norm, disp=0
            )
            pnoi_array[i_tj] = pnoi
            anoi_array[i_tj] = anoi

            # Calculate jitter
            jitter_rms = pnoi / (2 * np.pi * Fin_fit)
            meas_jitter[i_tj] = jitter_rms

            # Spectrum analysis
            ENoB, SNDR, SFDR, SNR, THD, pwr, NF, _ = spec_plot(
                read_data,
                label=1,
                harmonic=0,
                osr=1,
                nf_method=0
            )
            meas_SNDR[i_tj] = SNDR
            plt.close()  # Close the spectrum plot

            if (i_tj + 1) % 5 == 0:
                print(f'  [{i_tj+1}/{len(Tj_list)}] [test_jitter_load] [Tj={Tj*1e15:.2f}fs→{jitter_rms*1e15:.2f}fs] [SNDR={SNDR:.2f}dB] from {filename}')

        print()

        Fin_actual_mean = np.mean(actual_Fin)
        print(f'[Extracted Fin from data] = {Fin_actual_mean/1e6:.6f} MHz (nominal was {Fin_nominal/1e6:.6f} MHz)')

        # Plot results
        fig = plt.figure(figsize=(10, 6))
        ax1 = plt.gca()

        # Left y-axis: Jitter
        ax1.loglog(set_jitter, set_jitter, 'k--', linewidth=1.5, label='Set jitter')
        ax1.loglog(set_jitter, meas_jitter, 'bo', markersize=8, markerfacecolor='b', label='Calculated jitter')
        ax1.set_xlabel('Set jitter (seconds)', fontsize=18)
        ax1.set_ylabel('Jitter (seconds)', fontsize=18, color='b')
        ax1.set_ylim([min(set_jitter) * 0.5, max(set_jitter) * 2])
        ax1.tick_params(axis='y', labelcolor='b', labelsize=16)
        ax1.tick_params(axis='x', labelsize=16)
        ax1.grid(True)

        # Right y-axis: SNDR
        ax2 = ax1.twinx()
        ax2.semilogx(set_jitter, meas_SNDR, 's-', linewidth=2, markersize=8, color='r', label='SNDR')
        ax2.set_ylabel('SNDR (dB)', fontsize=18, color='r')
        ax2.set_ylim([0, 100])
        ax2.tick_params(axis='y', labelcolor='r', labelsize=16)

        plt.title(f'Jitter Analysis (Fin = {Fin_actual_mean/1e6:.1f} MHz)', fontsize=20)

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=16)

        # Save plot
        output_filename = f'jitter_analysis_Fin_{round(Fin_nominal/1e6)}MHz_python.png'
        save_fig(output_dir, output_filename)
        print(f'[Saved plot] -> [{output_dir / output_filename}]\n')

        # Save each variable to separate CSV (matching MATLAB format)
        # Create frequency-specific prefix for filenames
        freq_prefix = f'Fin_{round(Fin_nominal/1e6)}MHz'
        save_variable(output_dir, np.arange(1, len(Tj_list) + 1), f'{freq_prefix}_Tj_idx')
        save_variable(output_dir, set_jitter, f'{freq_prefix}_set_jitter_s')
        save_variable(output_dir, set_jitter * 1e15, f'{freq_prefix}_set_jitter_fs')
        save_variable(output_dir, meas_jitter, f'{freq_prefix}_meas_jitter_s')
        save_variable(output_dir, meas_jitter * 1e15, f'{freq_prefix}_meas_jitter_fs')
        save_variable(output_dir, pnoi_array, f'{freq_prefix}_pnoi_rad')
        save_variable(output_dir, anoi_array, f'{freq_prefix}_anoi')
        save_variable(output_dir, meas_SNDR, f'{freq_prefix}_SNDR_dB')
        save_variable(output_dir, actual_Fin, f'{freq_prefix}_actual_Fin_Hz')

    print('[test_jitter_load complete]')


if __name__ == "__main__":
    main()
