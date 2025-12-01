"""
exp01_sine_fit_spec_plot.py - Sine Fit and Spectrum Analysis Example

This example demonstrates:
- Loading example data from the package
- Fitting sine waves using sine_fit
- Plotting fitted results
- Analyzing spectrum using spec_plot

Usage:
    python -m adctoolbox.examples.exp01_sine_fit_spec_plot
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from adctoolbox.common import sine_fit
from adctoolbox.aout import spec_plot

plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True


def main():
    """Analyze sine wave data with fitting and spectrum analysis."""

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        "sinewave_jitter_400fs.csv",
        "sinewave_noise_270uV.csv",
    ]

    data_dir = Path(__file__).parent / "data"

    for idx, dataset_file in enumerate(datasets, 1):
        dataset_name = Path(dataset_file).stem
        

        data_path = data_dir / dataset_file
        raw_data = np.loadtxt(data_path, delimiter=',').flatten()
        print(f"\n[{idx}/{len(datasets)}] loaded [{len(raw_data)}] samples from [{dataset_file}]")

        data_fit, freq, mag, dc, phi = sine_fit(raw_data)
        print(f"  [sine_fit] Freq=[{freq:.6f}], Mag=[{mag:.6f}], DC=[{dc:.6f}], Phase=[{phi:.6f} rad ({np.degrees(phi):.2f}°)]")

        # Figure 1: Sine Fit
        period_samples = int(round(1.0 / freq)) if freq > 0 else len(raw_data)
        n_plot = min(max(int(period_samples * 3), 20), len(raw_data))

        fig1 = plt.figure(figsize=(10, 6))
        
        # A. Prepare Time Axes
        t_data = np.arange(n_plot)  # Discrete integer indices (0, 1, 2...)
        t_dense = np.linspace(0, n_plot - 1, n_plot * 50) # High-res float time for smooth curves

        # B. Original Data: the raw noisy input (Blue Dot)
        plt.plot(t_data, raw_data[:n_plot], 'bo', markersize=5, alpha=0.6, label='Original samples')
        
        # C. Function Output: discrete points returned by sine_fit() (Green X)
        # Should align perfectly with the red line at integer indices
        plt.plot(t_data, data_fit[:n_plot], 'gx', markersize=8, markeredgewidth=2, label='Tool output (data_fit)')


        # D. Smooth Reconstruction: calculated from scalar parameters (freq, mag, etc.) (Red Dashed)
        # This represents the "Theoretical Ideal"
        reconstructed_sine = mag * np.cos(2 * np.pi * freq * t_dense + phi) + dc
        plt.plot(t_dense, reconstructed_sine, 'r--', linewidth=2, alpha=0.8, label='Reconstructed with params)')

        # Formatting
        plt.title(f'Sine Fit Verification: {dataset_name}')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.legend(loc='best')
        plt.tight_layout()

        fig1_path = output_dir / f"exp01_sine_fit_{dataset_name}.png"
        plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
        print(f"  [save fig] -> [{fig1_path}]")
        plt.close()

        # Figure 2: Spectrum Analysis
        fig2 = plt.figure(figsize=(10, 6))
        ENoB, SNDR, SFDR, SNR, THD, pwr, NF, h = spec_plot(
            raw_data, Fs=1.0, harmonic=7, isPlot=1)
        print(f"  [spec_plot] ENoB=[{ENoB:.2f}], SNDR=[{SNDR:.2f} dB], SFDR=[{SFDR:.2f} dB], SNR=[{SNR:.2f} dB], THD=[{THD:.2f} dB], Power=[{pwr:.2f} dB], NF=[{-NF:.2f} dB]")

        fig2_path = output_dir / f"exp01_spectrum_{dataset_name}.png"
        plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
        print(f"  [save fig] -> [{fig2_path}]")
        plt.close()

        

if __name__ == "__main__":
    main()
