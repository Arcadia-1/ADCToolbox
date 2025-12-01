"""
exp01_sine_fit.py - Sine Fit and Spectrum Analysis Example

This example demonstrates:
- Loading example data from the package
- Fitting sine waves using sine_fit
- Plotting fitted results
- Analyzing spectrum using spec_plot

Usage:
    python -m adctoolbox.examples.exp01_sine_fit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from adctoolbox.common import sine_fit
from adctoolbox.aout import spec_plot

plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True


def main():
    """Analyze sine wave data with fitting and spectrum analysis."""

    print("\n" + "="*70)
    print("ADCToolbox - Sine Fit and Spectrum Analysis")
    print("="*70)

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        "sinewave_jitter_400fs.csv",
        "sinewave_noise_270uV.csv",
    ]

    data_dir = Path(__file__).parent / "data"

    for idx, dataset_file in enumerate(datasets, 1):
        dataset_name = Path(dataset_file).stem
        print(f"\n[{idx}/{len(datasets)}] {dataset_name}")

        data_path = data_dir / dataset_file
        raw_data = pd.read_csv(data_path, header=None).values.flatten()
        print(f"  [Loaded] {len(raw_data)} samples")

        data_fit, freq, mag, dc, phi = sine_fit(raw_data)
        print(f"  [Fit] Freq: {freq:.6f}, Mag: {mag:.6f}, DC: {dc:.6f}, Phase: {phi:.6f} rad ({np.degrees(phi):.2f}°)")

        # Figure 1: Sine Fit
        period_samples = int(round(1.0 / freq)) if freq > 0 else len(raw_data)
        n_plot = min(max(period_samples * 2, 50), len(raw_data))

        fig1 = plt.figure(figsize=(10, 6))
        t_data = np.arange(n_plot)
        plt.plot(t_data, raw_data[:n_plot], 'bo-', linewidth=1.5, markersize=4, label='Original Data')

        t_dense = np.linspace(0, n_plot - 1, n_plot * 50)
        fitted_sine = mag * np.cos(2 * np.pi * freq * t_dense + phi) + dc
        plt.plot(t_dense, fitted_sine, 'r--', linewidth=2, label='Fitted Sine')

        plt.title(f'Sine Fit: {dataset_file}')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()

        fig1_path = output_dir / f"{dataset_name}_sine_fit.png"
        plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
        plt.close(fig1)

        # Figure 2: Spectrum Analysis
        ENoB, SNDR, SFDR, SNR, THD, pwr, NF, h = spec_plot(raw_data, Fs=1.0, harmonic=7, isPlot=0)

        # Compute spectrum manually for plotting
        N = len(raw_data)
        Nd2 = N // 2
        freq_bins = np.arange(Nd2) / N

        # Apply windowing and FFT
        from scipy.signal import windows
        win = windows.hann(N, sym=False)
        tdata = raw_data - np.mean(raw_data)
        tdata = tdata * win / np.sqrt(np.mean(win**2))
        spec = np.abs(np.fft.fft(tdata))**2
        spec = spec[:Nd2] / (N**2) * 16
        spec[0] = 0  # Ignore DC
        spectrum_dB = 10 * np.log10(spec + 1e-12)

        fig2 = plt.figure(figsize=(10, 6))
        plt.plot(freq_bins, spectrum_dB, 'b-', linewidth=1.5)
        plt.title(f'Spectrum Analysis: {dataset_file}')
        plt.xlabel('Normalized Frequency')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True)
        plt.xlim([0, 0.5])

        metrics_text = f"SNR: {SNR:.2f} dB\nTHD: {THD:.2f} dB\nSFDR: {SFDR:.2f} dB"
        plt.text(0.98, 0.95, metrics_text,
                 transform=plt.gca().transAxes, verticalalignment='top',
                 horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        plt.tight_layout()

        fig2_path = output_dir / f"{dataset_name}_spectrum.png"
        plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
        plt.close(fig2)

        print(f"  [Spec] SNR: {SNR:.2f} dB, THD: {THD:.2f} dB, SFDR: {SFDR:.2f} dB")
        print(f"  [Saved] {fig1_path.name}, {fig2_path.name}")

    print("\n" + "="*70)
    print(f"[DONE] Processed {len(datasets)} datasets")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
