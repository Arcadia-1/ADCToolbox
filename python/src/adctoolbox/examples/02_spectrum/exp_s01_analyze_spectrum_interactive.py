"""Simplest spectrum analysis example with interactive plot."""
import numpy as np
import matplotlib.pyplot as plt
from adctoolbox import calc_coherent_freq, analyze_spectrum

N_fft = 2**13
Fs = 100e6
Fin_target = 12e6
Fin, Fin_bin = calc_coherent_freq(fs=Fs, fin_target=Fin_target, n_fft=N_fft)
A = 0.5
noise_rms = 50e-6

t = np.arange(N_fft) / Fs
signal = A * np.sin(2*np.pi*Fin*t) + np.random.randn(N_fft) * noise_rms

result = analyze_spectrum(signal, fs=Fs)

print(f"[ENOB] = {result['enob']:.2f} b, [SNDR] = {result['sndr_db']:.2f} dB, [SFDR] = {result['sfdr_db']:.2f} dB, [SNR] = {result['snr_db']:.2f} dB")

print("\n[Figure displayed - close the window to exit]")
plt.show()
