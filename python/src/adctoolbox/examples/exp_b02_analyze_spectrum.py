"""Frequency spectrum analysis - coherent vs non-coherent"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import calc_coherent_freq, analyze_spectrum

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N = 2**13
Fs = 1e6
t = np.arange(N) / Fs
A = 0.49
DC = 0.5
noise_rms = 50e-6

Fin_arbitrary = 123e3
Fin_coherent, bin = calc_coherent_freq(Fs, Fin_arbitrary, N)

signal_arbitrary = A * np.sin(2*np.pi*Fin_arbitrary*t) + DC + np.random.randn(N) * noise_rms
signal_coherent = A * np.sin(2*np.pi*Fin_coherent*t) + DC + np.random.randn(N) * noise_rms

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

plt.sca(ax1)
result1 = analyze_spectrum(signal_arbitrary, fs=Fs, harmonic=7, label=1)
ax1.set_title(f'Non-Coherent: Fin={Fin_arbitrary/1e3:.1f}kHz (spectral leakage!)')

plt.sca(ax2)
result2 = analyze_spectrum(signal_coherent, fs=Fs, harmonic=7, label=1)
ax2.set_title(f'Coherent: Fin={Fin_coherent/1e3:.3f}kHz (Bin {bin})')

print(f"[Spectrum Comparison]")
print(f"[Fin = {Fin_arbitrary/1e3:6.3f} kHz] [ENoB = {result1['enob']:5.2f}] [SNDR = {result1['sndr_db']:.2f} dB] [SFDR = {result1['sfdr_db']:6.2f} dB] [SNR = {result1['snr_db']:.2f} dB]  (Non-coherent sampling)")
print(f"[Fin = {Fin_coherent/1e3:6.3f} kHz] [ENoB = {result2['enob']:5.2f}] [SNDR = {result2['sndr_db']:.2f} dB] [SFDR = {result2['sfdr_db']:6.2f} dB] [SNR = {result2['snr_db']:.2f} dB]  (Coherent sampling)")

plt.tight_layout()
fig_path = (output_dir / 'exp_b02_analyze_spectrum.png').resolve()
print(f"[Save fig] -> [{fig_path}]\n")
plt.savefig(fig_path, dpi=150)
plt.close()