"""
Basic demo: Spectrum analysis with interactive plot.

This script demonstrates using the analyze_spectrum function for performing standard FFT analysis and displaying the interactive plot.
"""
import numpy as np
from adctoolbox import analyze_spectrum

N_fft = 2**13
Fs = 100e6
Fin = 123/N_fft * Fs  # Coherent frequency
t = np.arange(N_fft) / Fs
signal = 0.5 * np.sin(2*np.pi*Fin*t) + np.random.randn(N_fft) * 50e-6

result = analyze_spectrum(signal, fs=Fs)

print(f"[analyze_spectrum] ENOB = [{result['enob']:.2f} b], SNDR = [{result['sndr_db']:.2f} dB], SFDR = [{result['sfdr_db']:.2f} dB], SNR = [{result['snr_db']:.2f} dB]")
