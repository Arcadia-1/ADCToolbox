"""
Basic demo: Comparison of Coherent vs. Non-Coherent Sampling.

This script demonstrates the critical effect of spectral leakage by analyzing 
the same signal with an arbitrary (non-coherent) frequency and a calculated 
coherent frequency. It shows how coherent sampling reduces leakage.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import calc_coherent_freq, calculate_spectrum_metrics, plot_spectrum

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N_fft = 2**13
Fs = 100e6
A = 0.5
noise_rms = 50e-6

Fin_arbitrary = 10e6
Fin_coherent, Fin_bin = calc_coherent_freq(Fs, Fin_arbitrary, N_fft)

print(f"[Analysis Parameters] N_fft = {N_fft}, Fs = {Fs/1e6:.2f} MHz, Fin_arb = {Fin_arbitrary/1e6:.3f} MHz, Fin_coherent = {Fin_coherent/1e6:.3f} MHz (Bin {Fin_bin})\n")

t = np.arange(N_fft) / Fs
signal_arbitrary = A * np.sin(2*np.pi*Fin_arbitrary*t)  + np.random.randn(N_fft) * noise_rms
signal_coherent = A * np.sin(2*np.pi*Fin_coherent*t) + np.random.randn(N_fft) * noise_rms

metrics1, plot_data1 = calculate_spectrum_metrics(signal_arbitrary, fs=Fs, harmonic=7)
print(f"[Non-coherent] [ENOB] = {metrics1['enob']:5.2f} b, [SNDR] = {metrics1['sndr_db']:5.2f} dB, [SFDR] = {metrics1['sfdr_db']:6.2f} dB")

metrics2, plot_data2 = calculate_spectrum_metrics(signal_coherent, fs=Fs, harmonic=7)
print(f"[Coherent]     [ENOB] = {metrics2['enob']:5.2f} b, [SNDR] = {metrics2['sndr_db']:5.2f} dB, [SFDR] = {metrics2['sfdr_db']:6.2f} dB\n")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

plt.sca(ax1)
plot_spectrum(metrics1, plot_data1, harmonic=7, freq_scale='linear', show_label=True, ax=ax1)
ax1.set_title(f'Non-Coherent: Fin={Fin_arbitrary/1e3:.1f}kHz (spectral leakage!)')

plt.sca(ax2)
plot_spectrum(metrics2, plot_data2, harmonic=7, freq_scale='linear', show_label=True, ax=ax2)
ax2.set_title(f'Coherent: Fin={Fin_coherent/1e3:.3f}kHz (Bin {bin})')

plt.tight_layout()
fig_path = (output_dir / 'exp_b03_analyze_spectrum_manual.png').resolve()
print(f"[Save fig] -> [{fig_path}]\n")
plt.savefig(fig_path, dpi=150)
plt.close()