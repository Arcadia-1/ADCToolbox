"""
Basic demo: Spectrum analysis with interactive plot.

This script demonstrates using the analyze_spectrum function for performing standard FFT analysis and displaying the interactive plot.
"""
import numpy as np
import matplotlib.pyplot as plt
from adctoolbox import analyze_spectrum, calculate_coherent_freq, calculate_snr_from_amplitude, snr_to_nsd

N_fft = 2**13
Fs = 100e6
Fin_target = 12e6
Fin, Fin_bin = calculate_coherent_freq(fs=Fs, fin_target=Fin_target, n_fft=N_fft)
A = 0.5
noise_rms = 50e-6

snr_ref = calculate_snr_from_amplitude(sig_amplitude=A, noise_amplitude=noise_rms)
nsd_ref = snr_to_nsd(snr_ref, fs=Fs, osr=1)
print(f"[Setting] N=[{N_fft}], Fs=[{Fs/1e6:.1f} MHz], Fin=[{Fin/1e6:.1f} MHz] (Bin=[{Fin_bin}]) | [Theory] SNR=[{snr_ref:.2f} dB], NSD=[{nsd_ref:.2f} dBFS/Hz]")

t = np.arange(N_fft) / Fs
signal = A * np.sin(2*np.pi*Fin*t) + np.random.randn(N_fft) * noise_rms

result = analyze_spectrum(signal, fs=Fs)

print(f"[analyze_spectrum] ENOB=[{result['enob']:.2f} b], SNDR=[{result['sndr_db']:.2f} dB], SFDR=[{result['sfdr_db']:.2f} dB], SNR=[{result['snr_db']:.2f} dB], NSD=[{result['nsd_dbfs_hz']:.2f} dBFS/Hz]")

print("\n[Figure displayed - close the window to exit]")
plt.show()
