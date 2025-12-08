"""
Basic demo: Spectrum analysis and figure saving.

This script demonstrates using the analyze_spectrum function for performing standard FFT analysis and saving the figure directly to the output directory.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import analyze_spectrum, calc_coherent_freq

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N_fft = 2**13
Fs = 100e6
Fin_target = 12e6
Fin, Fin_bin = calc_coherent_freq(fs=Fs, fin_target=Fin_target, n_fft=N_fft)
A = 0.5
noise_rms = 200e-6

print(f"[Analysis Parameters] N = {N_fft}, Fs = {Fs/1e6:.2f} MHz, Fin = {Fin/1e6:.4f} MHz (Bin = {Fin_bin})")

t = np.arange(N_fft) / Fs
signal = A * np.sin(2*np.pi*Fin*t) + np.random.randn(N_fft) * noise_rms

result = analyze_spectrum(signal, fs=Fs)

print(f"[ENOB] = {result['enob']:.2f} b, [SNDR] = {result['sndr_db']:.2f} dB, [SFDR] = {result['sfdr_db']:.2f} dB, [SNR] = {result['snr_db']:.2f} dB")

fig_path = (output_dir / 'exp_b02_simple_spectrum.png').resolve()
print(f"\n[Save fig] -> [{fig_path}]\n")
plt.savefig(fig_path, dpi=150)
plt.close()