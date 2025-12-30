"""
Basic demo: Spectrum analysis with interactive plot.

This script demonstrates using the analyze_spectrum function for performing standard FFT analysis and displaying the interactive plot.
"""
import numpy as np
import matplotlib.pyplot as plt
from adctoolbox import analyze_spectrum, find_coherent_frequency, amplitudes_to_snr, snr_to_nsd

N_fft = 2**13
Fs = 100e6
Fin_target = 12e6
Fin, Fin_bin = find_coherent_frequency(fs=Fs, fin_target=Fin_target, n_fft=N_fft)
A = 0.5
noise_rms = 50e-6

snr_ref = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=noise_rms)
nsd_ref = snr_to_nsd(snr_ref, fs=Fs, osr=1)
print(f"[Sinewave] Fs=[{Fs/1e6:.2f} MHz], Fin=[{Fin/1e6:.2f} MHz], Bin/N=[{Fin_bin}/{N_fft}], A=[{A:.3f} Vpeak]")
print(f"[Nonideal] Noise RMS=[{noise_rms*1e6:.2f} uVrms], Theoretical SNR=[{snr_ref:.2f} dB], Theoretical NSD=[{nsd_ref:.2f} dBFS/Hz]\n")

t = np.arange(N_fft) / Fs
signal = A * np.sin(2*np.pi*Fin*t) + np.random.randn(N_fft) * noise_rms

result = analyze_spectrum(signal, fs=Fs)

print(f"[analyze_spectrum] ENoB=[{result['enob']:.2f} b], SNDR=[{result['sndr_dbc']:.2f} dB], SFDR=[{result['sfdr_dbc']:.2f} dB], SNR=[{result['snr_dbc']:.2f} dB], NSD=[{result['nsd_dbfs_hz']:.2f} dBFS/Hz]")

print("\n[Figure displayed - close the window to exit]")
plt.show()
