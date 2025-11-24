"""
Jitter sweep test - matches MATLAB test_jitter.m exactly.

Golden reference: ADCToolbox_test/test_jitter.m (VERIFIED!)

This script:
1. Generates sine waves with varying jitter (logspace(-15, -12, 20))
2. Measures jitter using errHistSine (pnoi output)
3. Measures SNDR using spec_plot
4. Plots and saves results to ADCToolbox_example_output/jitter_sweep/
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from ADCToolbox_Python.errHistSine import errHistSine
from ADCToolbox_Python.spec_plot import spec_plot
from ADCToolbox_Python.findBin import find_bin


def main():
    """Main jitter sweep test - matches MATLAB test_jitter.m exactly."""

    print("="*80)
    print("Jitter Sweep Test")
    print("Based on MATLAB golden reference: ADCToolbox_test/test_jitter.m")
    print("="*80)
    print()

    # Constants (MATLAB lines 3-11)
    N = 2**14
    Fs = 10e9
    J = find_bin(Fs, 1000e6, N)
    Fin = J/N * Fs

    A = 0.49
    offset = 0.5
    amp_noise = 0.00001

    print(f"[N] = {N}")
    print(f"[Fs] = {Fs/1e9:.2f} GHz")
    print(f"[J] = {J}")
    print(f"[Fin] = {Fin/1e9:.4f} GHz")
    print()

    # Jitter list (MATLAB line 14)
    Tj_list = np.logspace(-15, -12, 20)

    # Number of random trials (MATLAB line 17)
    N_random = 5

    # Results arrays (MATLAB lines 20-22)
    meas_jitter_new = np.zeros((len(Tj_list), N_random))
    meas_SNDR = np.zeros((len(Tj_list), N_random))

    print(f"[Jitter range] = {Tj_list[0]*1e15:.2f} fs to {Tj_list[-1]*1e15:.2f} fs")
    print(f"[Number of jitter points] = {len(Tj_list)}")
    print(f"[Trials per jitter] = {N_random}")
    print()
    print("-"*80)

    # Main loop (MATLAB lines 24-58)
    for i in range(len(Tj_list)):
        Tj = Tj_list[i]

        for k in range(N_random):
            # Generate jittered signal (MATLAB lines 31-42)
            Ts = 1 / Fs
            theta = 2 * np.pi * Fin * np.arange(N) * Ts

            # Convert jitter(sec) -> phase jitter(rad)
            phase_noise_rms = 2 * np.pi * Fin * Tj

            # Random jitter
            phase_jitter = np.random.randn(N) * phase_noise_rms

            # Jittered signal
            data = np.sin(theta + phase_jitter) * A + offset + np.random.randn(N) * amp_noise

            # Extract jitter by errHistSine (MATLAB line 45)
            # MATLAB: [emean, erms, phase_code, anoi, pnoi] = errHistSine(data, 99, J/N, 0);
            emean, erms, phase_code, anoi, pnoi, err, xx = errHistSine(data, bin=99, fin=J/N, disp=0)

            # Convert pnoi to jitter (MATLAB line 47)
            jitter_rms_new = pnoi / (2*np.pi*Fin)
            meas_jitter_new[i, k] = jitter_rms_new

            # Measure SNDR (MATLAB line 50)
            ENoB, SNDR, SFDR, SNR, THD, pwr, NF, h = spec_plot(
                data,
                label=1,
                harmonic=0,
                winType=1,  # hann window
                OSR=1,
                isPlot=0
            )

            meas_SNDR[i, k] = SNDR

            # Print progress (MATLAB lines 54-55)
            print(f"[Tj]={Tj*1e15:8.2f}fs, [trial {k+1}] -> [ENoB={ENoB:.2f}] [measured jitter2] {jitter_rms_new*1e15:.2f}fs")

    print("-"*80)
    print()

    # Compute average of N_random runs (MATLAB lines 60-62)
    avg_meas_jitter_new = np.mean(meas_jitter_new, axis=1)
    avg_meas_SNDR = np.mean(meas_SNDR, axis=1)

    # Plot (MATLAB lines 64-95)
    fig = plt.figure(figsize=(10, 7))

    # Left axis: jitter (MATLAB lines 66-71)
    ax1 = fig.add_subplot(111)
    ax1.loglog(Tj_list, Tj_list, 'k--', linewidth=1.5, label='set jitter')
    ax1.loglog(Tj_list, avg_meas_jitter_new, 'bs-', linewidth=2, markersize=8, label='Calculated jitter')

    ax1.set_ylabel('Calculated jitter (s)', fontsize=18, color='C0')
    ax1.tick_params(axis='y', labelcolor='C0')
    ax1.tick_params(labelsize=16)

    # Right axis: SNDR (MATLAB lines 73-77)
    ax2 = ax1.twinx()
    ax2.semilogx(Tj_list, avg_meas_SNDR, 's-', linewidth=2, markersize=8, color='C1', label='SNDR')
    ax2.set_ylabel('SNDR (dB)', fontsize=18, color='C1')
    ax2.set_ylim([0, 100])
    ax2.tick_params(axis='y', labelcolor='C1')
    ax2.tick_params(labelsize=16)

    # Shared x axis (MATLAB line 80)
    ax1.set_xlabel('Set jitter (seconds)', fontsize=18)

    # Title (MATLAB line 82)
    ax1.set_title(f'Jitter and SNDR (Fin = {Fin/1e9:.2f}GHz)', fontsize=20)

    # Combined legend (MATLAB line 84) - "southeast" -> "lower right" in matplotlib
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=16)

    # Grid (MATLAB line 86)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure (MATLAB lines 90-101)
    # Go to project root, then to output directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    outputdir = os.path.join(project_root, "ADCToolbox_example_output")
    subdir_path = os.path.join(outputdir, "jitter_sweep")
    os.makedirs(subdir_path, exist_ok=True)

    # MATLAB format: jitter_sweep_Fin_200MHz_matlab.png
    # Python format: jitter_sweep_Fin_200MHz_python.png
    output_filename = f'jitter_sweep_Fin_{int(np.round(Fin/1e6))}MHz_python.png'
    output_filepath = os.path.join(subdir_path, output_filename)

    plt.savefig(output_filepath, dpi=150, bbox_inches='tight')
    print(f'[Saved image] -> [{output_filepath}]')
    print()

    plt.close()

    # Print summary
    print("="*80)
    print("Summary:")
    print(f"  Jitter range: {Tj_list[0]*1e15:.2f} fs to {Tj_list[-1]*1e15:.2f} fs")
    print(f"  Number of points: {len(Tj_list)}")
    print(f"  Trials per point: {N_random}")
    print(f"  Output: {output_filepath}")
    print("="*80)


if __name__ == "__main__":
    # Set random seed for reproducibility (optional)
    np.random.seed(42)

    main()
