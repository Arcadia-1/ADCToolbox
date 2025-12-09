"""
Basic demo: Spectrum analysis and figure saving.

This script demonstrates using the analyze_spectrum function for performing standard FFT analysis and saving the figure directly to the output directory.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import analyze_spectrum, calculate_coherent_freq, calculate_snr_from_amplitude, snr_to_nsd

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N_fft = 2**13
Fs = 100e6
Fin_target = 12e6
Fin, Fin_bin = calculate_coherent_freq(fs=Fs, fin_target=Fin_target, n_fft=N_fft)
A = 0.5
noise_rms = 200e-6

snr_ref = calculate_snr_from_amplitude(sig_amplitude=A, noise_amplitude=noise_rms)
nsd_ref = snr_to_nsd(snr_ref, fs=Fs, osr=1)
print(f"[Sinewave] Fs=[{Fs/1e6:.2f} MHz], Fin=[{Fin/1e6:.2f} MHz], Bin/N=[{Fin_bin}/{N_fft}], A=[{A:.3f} Vpeak]")
print(f"[Nonideal] Noise RMS=[{noise_rms*1e6:.2f} uVrms], Theoretical SNR=[{snr_ref:.2f} dB], Theoretical NSD=[{nsd_ref:.2f} dBFS/Hz]\n")

t = np.arange(N_fft) / Fs
signal = A * np.sin(2*np.pi*Fin*t) + np.random.randn(N_fft) * noise_rms

result = analyze_spectrum(signal, fs=Fs)

print(f"[analyze_spectrum] ENoB=[{result['enob']:5.2f} b], SNDR=[{result['sndr_db']:6.2f} dB], SFDR=[{result['sfdr_db']:6.2f} dB], SNR=[{result['snr_db']:6.2f} dB], NSD=[{result['nsd_dbfs_hz']:7.2f} dBFS/Hz]")

fig_path = (output_dir / 'exp_s02_analyze_spectrum_savefig.png').resolve()
print(f"\n[Save fig] -> [{fig_path}]\n")
plt.savefig(fig_path, dpi=150)
plt.close()