"""
Jitter sweep test for ADC analysis.

Based on MATLAB golden reference: matlab_reference/test_jitter.m

This testbench:
1. Generates sine waves with varying jitter levels (100fs to 1ns)
2. Calculates jitter using calculate_jitter() function
3. Measures SNDR using spec_plot()
4. Plots results matching MATLAB golden reference format
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from ADC_Toolbox_Python.calculate_jitter import calculate_jitter
from ADC_Toolbox_Python.spec_plot import spec_plot
from ADC_Toolbox_Python.findBin import find_bin


def generate_jittered_signal(N, Fs, Fin, Tj, A=0.49, offset=0.5, amp_noise=0.00001):
    """
    Generate a sine wave with jitter.

    Matches MATLAB code lines 30-41:
        Ts = 1 / Fs;
        theta = 2 * pi * Fin * (0:N - 1) * Ts;
        phase_noise_rms = 2 * pi * Fin * Tj;
        phase_jitter = randn(1, N) * phase_noise_rms;
        data = sin(theta + phase_jitter) * A + offset + randn(1, N) * amp_noise;

    Args:
        N: Number of samples
        Fs: Sampling frequency (Hz)
        Fin: Input frequency (Hz)
        Tj: Jitter RMS (seconds)
        A: Signal amplitude
        offset: DC offset
        amp_noise: Amplitude noise

    Returns:
        data: Noisy sine wave with jitter
    """
    # Ideal phase
    Ts = 1 / Fs
    theta = 2 * np.pi * Fin * np.arange(N) * Ts

    # Convert jitter (sec) -> phase jitter (rad)
    phase_noise_rms = 2 * np.pi * Fin * Tj

    # Random jitter
    phase_jitter = np.random.randn(N) * phase_noise_rms

    # Jittered signal
    data = np.sin(theta + phase_jitter) * A + offset + np.random.randn(N) * amp_noise

    return data


def run_jitter_sweep(Fin_Hz, output_dir=None, N_random=1):
    """
    Run jitter sweep for a specific input frequency.

    Args:
        Fin_Hz: Input frequency in Hz (e.g., 100e6, 1.05e9, 3.0e9)
        output_dir: Output directory for plots
        N_random: Number of random trials per jitter value

    Returns:
        results: Dictionary with sweep results
    """
    print("=" * 80)
    print(f"Jitter Sweep Test - Fin = {Fin_Hz/1e9:.2f} GHz")
    print("=" * 80)

    # Constants (matching MATLAB reference lines 4-7)
    N = 2**14  # 16384 samples
    Fs = 10e9  # 10 GHz sampling rate

    # Find coherent bin
    J = find_bin(Fs, Fin_Hz, N)
    Fin = J / N * Fs  # Actual coherent frequency
    fin_norm = J / N  # Normalized frequency

    print(f"Sampling frequency: {Fs/1e9:.2f} GHz")
    print(f"Input frequency: {Fin/1e9:.4f} GHz (bin {J})")
    print(f"Number of samples: {N}")
    print(f"Number of trials per jitter: {N_random}")

    # Signal parameters (matching MATLAB lines 9-11)
    A = 0.49
    offset = 0.5
    amp_noise = 0.00001

    # Jitter list (100fs to 1ns, 30 points) - matching MATLAB line 14
    Tj_list = np.logspace(-15, -12, 30)

    # Results arrays (matching MATLAB lines 20-21)
    meas_jitter = np.zeros((len(Tj_list), N_random))
    meas_SNDR = np.zeros((len(Tj_list), N_random))

    # Run sweep (matching MATLAB lines 24-65)
    print("\nRunning sweep...")
    print("-" * 80)

    for i, Tj in enumerate(Tj_list):
        for k in range(N_random):
            # Generate jittered signal
            data = generate_jittered_signal(N, Fs, Fin, Tj, A, offset, amp_noise)

            # Calculate jitter - must pass Fin_Hz!
            jitter_rms = calculate_jitter(data, fin=fin_norm, Fin_Hz=Fin)
            meas_jitter[i, k] = jitter_rms

            # Measure SNDR
            ENoB, SNDR, SFDR, SNR, THD, pwr, NF, h = spec_plot(
                data,
                label=1,
                harmonic=0,
                winType=1,  # 1 = hann window
                OSR=1,
                isPlot=0
            )
            meas_SNDR[i, k] = SNDR

            # Print progress (matching MATLAB lines 62-63)
            print(f"[Tj]={Tj*1e15:8.2f}fs, [trial {k+1}] -> [ENoB={ENoB:.2f}] "
                  f"[measured jitter] {jitter_rms*1e15:.2f}fs")

    # Compute average of N_random runs (matching MATLAB lines 69-70)
    avg_meas_jitter = np.mean(meas_jitter, axis=1)
    avg_meas_SNDR = np.mean(meas_SNDR, axis=1)

    print("-" * 80)
    print("Sweep complete!")

    # Plot results (matching MATLAB format lines 72-96)
    fig = plt.figure(figsize=(10, 7))

    # Left axis: jitter (matching MATLAB lines 75-79)
    ax1 = fig.add_subplot(111)
    ax1.loglog(Tj_list, Tj_list, 'k--', linewidth=1.5, label='Set jitter')
    ax1.loglog(Tj_list, avg_meas_jitter, 'o-', linewidth=2,
               markersize=8, label='Calculated jitter')
    ax1.set_xlabel('Set jitter (seconds)', fontsize=18)
    ax1.set_ylabel('Calculated jitter (seconds)', fontsize=18, color='C0')
    ax1.tick_params(axis='y', labelcolor='C0')
    ax1.tick_params(labelsize=16)

    # Right axis: SNDR (matching MATLAB lines 82-85)
    ax2 = ax1.twinx()
    ax2.semilogx(Tj_list, avg_meas_SNDR, 's-', linewidth=2,
                 markersize=8, color='C1', label='Measured SNDR')
    ax2.set_ylabel('SNDR (dB)', fontsize=18, color='C1')
    ax2.set_ylim([0, 100])
    ax2.tick_params(axis='y', labelcolor='C1')
    ax2.tick_params(labelsize=16)

    # Title (matching MATLAB line 90)
    ax1.set_title(f'Jitter and SNDR (Fin = {Fin/1e9:.2f}GHz)', fontsize=20)

    # Combined legend (matching MATLAB lines 92-93)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=16)

    # Grid (matching MATLAB line 95)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure (matching MATLAB lines 103-108)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Format filename like MATLAB: jitter_sweep_of_Fin_0P10GHz_python.png
        Fin_GHz = Fin / 1e9
        s = f'{Fin_GHz:.2f}'
        s = s.replace('.', 'P')
        Fin_str = f'Fin_{s}GHz'
        filename = f'jitter_sweep_of_{Fin_str}_python.png'
        filepath = os.path.join(output_dir, filename)

        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved: {filepath}")

    plt.close()

    # Return results
    results = {
        'Fin_Hz': Fin,
        'Tj_list': Tj_list,
        'avg_meas_jitter': avg_meas_jitter,
        'avg_meas_SNDR': avg_meas_SNDR,
        'meas_jitter_all': meas_jitter,
        'meas_SNDR_all': meas_SNDR
    }

    return results


def main():
    """Main test function."""
    print("=" * 80)
    print("Jitter Sweep Test Suite")
    print("Based on: matlab_reference/test_jitter.m")
    print("=" * 80)
    print()

    # Output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output", "jitter_sweep")

    # Test frequencies (matching golden references)
    frequencies = [
        0.10e9,   # 100 MHz -> jitter_sweep_of_Fin_0P10GHz
        1.05e9,   # 1.05 GHz -> jitter_sweep_of_Fin_1P05GHz
        3.00e9,   # 3.00 GHz -> jitter_sweep_of_Fin_3P00GHz
    ]

    all_results = {}

    for Fin_Hz in frequencies:
        results = run_jitter_sweep(Fin_Hz, output_dir=output_dir, N_random=1)
        all_results[Fin_Hz] = results
        print()

    print("=" * 80)
    print("All tests complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    return all_results


if __name__ == "__main__":
    results = main()
