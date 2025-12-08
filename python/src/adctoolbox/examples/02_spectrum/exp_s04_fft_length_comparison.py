"""Demonstrate effect of FFT length on spectrum and noise floor.

This example shows how increasing N (FFT length) affects:
1. Frequency resolution (bin width)
2. Noise floor per bin (decreases with N)
3. Overall spectrum quality
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import analyze_spectrum, calculate_coherent_freq, calculate_snr_from_amplitude, snr_to_nsd

# Output directory
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Common parameters
Fs = 100e6
Fin_target = 12e6
A = 0.5
noise_rms = 200e-6

# Test different FFT lengths: 64, 128, 256, 1024, 8192, 16384, 65536, 262144
N_values = [2**6, 2**7, 2**8, 2**10, 2**13, 2**14, 2**16, 2**18]
n_cases = len(N_values)

# Theoretical SNR and NSD
snr_ref = calculate_snr_from_amplitude(sig_amplitude=A, noise_amplitude=noise_rms)
nsd_ref = snr_to_nsd(snr_ref, fs=Fs, osr=1)

print(f"[Parameters] Fs = [{Fs/1e6:.1f} MHz], Fin_target = [{Fin_target/1e6:.1f} MHz], Noise RMS = [{noise_rms*1e6:.1f} uV] | [Theoretical] SNR = [{snr_ref:.2f} dB], NSD = [{nsd_ref:.2f} dBFS/Hz]")

# Calculate grid dimensions and figure size
n_cols = 4
n_rows = (n_cases + n_cols - 1) // n_cols  # Ceiling division
fig_width = n_cols * 6  # 6 inches per column
fig_height = n_rows * 5  # 5 inches per row

# Create subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
axes = axes.flatten()

for idx, N_fft in enumerate(N_values):
    # Calculate coherent frequency for this N
    Fin, Fin_bin = calculate_coherent_freq(fs=Fs, fin_target=Fin_target, n_fft=N_fft)

    # Generate signal
    t = np.arange(N_fft) / Fs
    signal = A * np.sin(2*np.pi*Fin*t) + np.random.randn(N_fft) * noise_rms

    # Analyze spectrum
    result = analyze_spectrum(signal, fs=Fs, ax=axes[idx], show_plot=True)

    # Set y-axis limits to [-120, 0]
    axes[idx].set_ylim([-120, 0])

    # Calculate bin width
    bin_width = Fs / N_fft

    # Print results
    print(f"[N = {N_fft:5d} (2^{int(np.log2(N_fft)):2d})] [Bin width = {bin_width/1e3:7.3f} kHz] ENOB = [{result['enob']:.2f} b], SNDR = [{result['sndr_db']:.2f} dB], SNR = [{result['snr_db']:.2f} dB], NSD = [{result['nsd_dbfs_hz']:.2f} dBFS/Hz]")

    # Update subplot title
    axes[idx].set_title(f"N = {N_fft} (2^{int(np.log2(N_fft))}), Bin Width = {bin_width/1e3:.3f} kHz")

# Remove empty subplots if any
for idx in range(n_cases, len(axes)):
    axes[idx].remove()

plt.tight_layout()
fig_path = (output_dir / 'exp_s04_fft_length_comparison.png').resolve()
print(f"\n[Save fig] -> [{fig_path}]\n")
plt.savefig(fig_path, dpi=150)
plt.close()