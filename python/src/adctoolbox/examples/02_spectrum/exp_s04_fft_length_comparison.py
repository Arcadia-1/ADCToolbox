"""Demonstrate effect of FFT length on spectrum and noise floor.

This example shows how increasing N (FFT length) affects:
1. Frequency resolution (bin width)
2. Noise floor per bin (decreases with N)
3. Overall spectrum quality
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import analyze_spectrum, find_coherent_frequency, amplitudes_to_snr, snr_to_nsd

# Output directory
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Common parameters
Fs = 100e6
Fin_target = 12e6
A = 0.5
noise_rms = 200e-6

# Theoretical SNR and NSD
snr_ref = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=noise_rms)
nsd_ref = snr_to_nsd(snr_ref, fs=Fs, osr=1)
print(f"[Sinewave] Fs=[{Fs/1e6:.2f} MHz], Fin_target=[{Fin_target/1e6:.2f} MHz], A=[{A:.3f} Vpeak]")
print(f"[Nonideal] Noise RMS=[{noise_rms*1e6:.2f} uVrms], Theoretical SNR=[{snr_ref:.2f} dB], Theoretical NSD=[{nsd_ref:.2f} dBFS/Hz]\n")

# Test different FFT lengths
N_values = [2**7, 2**10, 2**13, 2**16]

n_cols = len(N_values)
fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 6, 5))
axes = axes.flatten()

for idx, N_fft in enumerate(N_values):
    # Calculate coherent frequency for this N
    Fin, Fin_bin = find_coherent_frequency(fs=Fs, fin_target=Fin_target, n_fft=N_fft)

    # Generate signal
    t = np.arange(N_fft) / Fs
    signal = A * np.sin(2*np.pi*Fin*t) + np.random.randn(N_fft) * noise_rms

    # Analyze spectrum
    plt.sca(axes[idx])
    result = analyze_spectrum(signal, fs=Fs)
    axes[idx].set_ylim([-120, 0])
    axes[idx].set_title(f"N = {N_fft}")

    bin_width = Fs / N_fft

    print(f"[N={N_fft:8d} (2^{int(np.log2(N_fft)):2d})] [1 Bin = {bin_width/1e3:8.3f} kHz] ENoB=[{result['enob']:5.2f} b], SNDR=[{result['sndr_db']:6.2f} dB], SFDR=[{result['sfdr_db']:6.2f} dB], SNR=[{result['snr_db']:6.2f} dB], NSD=[{result['nsd_dbfs_hz']:7.2f} dBFS/Hz]")


plt.tight_layout()
fig_path = (output_dir / 'exp_s04_fft_length_comparison.png').resolve()
print(f"\n[Save fig] -> [{fig_path}]\n")
plt.savefig(fig_path, dpi=150)
plt.close()