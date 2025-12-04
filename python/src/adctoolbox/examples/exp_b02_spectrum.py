"""Frequency spectrum analysis - coherent vs non-coherent"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_bin, spec_plot

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N = 2**13
Fs = 1e6
t = np.arange(N) / Fs
A = 0.49
DC = 0.5
noise_rms = 50e-6

Fin_arbitrary = 123e3
J = find_bin(Fs, Fin_arbitrary, N)
Fin_coherent = J / N * Fs

signal_arbitrary = A * np.sin(2*np.pi*Fin_arbitrary*t) + DC + np.random.randn(N) * noise_rms
signal_coherent = A * np.sin(2*np.pi*Fin_coherent*t) + DC + np.random.randn(N) * noise_rms

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

plt.sca(ax1)
enob1, sndr1, sfdr1, snr1, thd1, pwr1, nf1, nsd1, h1 = spec_plot(signal_arbitrary, fs=Fs, harmonic=7, label=1)
ax1.set_title(f'Non-Coherent: Fin={Fin_arbitrary/1e3:.1f}kHz (spectral leakage!)')

plt.sca(ax2)
enob2, sndr2, sfdr2, snr2, thd2, pwr2, nf2, nsd2, h2 = spec_plot(signal_coherent, fs=Fs, harmonic=7, label=1)
ax2.set_title(f'Coherent: Fin={Fin_coherent/1e3:.3f}kHz (Bin {J})')

print(f"[Spectrum Comparison]")
print(f"[Fin = {Fin_arbitrary/1e3:6.3f} kHz] [ENoB = {enob1:5.2f}] [SNDR = {sndr1:.2f} dB] [SFDR = {sfdr1:6.2f} dB] [SNR = {snr1:.2f} dB]  (Non-coherent sampling)")
print(f"[Fin = {Fin_coherent/1e3:6.3f} kHz] [ENoB = {enob2:5.2f}] [SNDR = {sndr2:.2f} dB] [SFDR = {sfdr2:6.2f} dB] [SNR = {snr2:.2f} dB]  (Coherent sampling)")

plt.tight_layout()
fig_path = (output_dir / 'exp_b02_spectrum.png').resolve()
print(f"[Save fig] -> [{fig_path}]\n")
plt.savefig(fig_path, dpi=150)
plt.close()