"""
Basic demo: Comparison of Coherent vs. Non-Coherent Sampling.

This script demonstrates the critical effect of spectral leakage by analyzing
the same signal with an arbitrary (non-coherent) frequency and a calculated
coherent frequency. It shows how coherent sampling reduces leakage.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import calculate_coherent_freq, analyze_spectrum, calculate_snr_from_amplitude, snr_to_nsd

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N_fft = 2**13
Fs = 100e6
A = 0.5
noise_rms = 50e-6

Fin_arbitrary = 10e6
Fin_coherent, Fin_bin = calculate_coherent_freq(Fs, Fin_arbitrary, N_fft)

snr_ref = calculate_snr_from_amplitude(sig_amplitude=A, noise_amplitude=noise_rms)
nsd_ref = snr_to_nsd(snr_ref, fs=Fs, osr=1)
print(f"[Parameters] N = [{N_fft}], Fs = [{Fs/1e6:.1f} MHz], Fin_arb = [{Fin_arbitrary/1e6:.1f} MHz], Fin_coh = [{Fin_coherent/1e6:.3f} MHz] (Bin = [{Fin_bin}]) | [Theoretical] SNR = [{snr_ref:.2f} dB], NSD = [{nsd_ref:.2f} dBFS/Hz]\n")

t = np.arange(N_fft) / Fs
signal_arbitrary = A * np.sin(2*np.pi*Fin_arbitrary*t)  + np.random.randn(N_fft) * noise_rms
signal_coherent = A * np.sin(2*np.pi*Fin_coherent*t) + np.random.randn(N_fft) * noise_rms

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

metrics1 = analyze_spectrum(signal_arbitrary, fs=Fs, plot_harmonics_up_to=7, ax=ax1)
print(f"[Non-coherent] [ENOB] = {metrics1['enob']:5.2f} b, [SNDR] = {metrics1['sndr_db']:5.2f} dB, [SFDR] = {metrics1['sfdr_db']:6.2f} dB")

metrics2 = analyze_spectrum(signal_coherent, fs=Fs, plot_harmonics_up_to=7, ax=ax2)
print(f"[Coherent]     [ENOB] = {metrics2['enob']:5.2f} b, [SNDR] = {metrics2['sndr_db']:5.2f} dB, [SFDR] = {metrics2['sfdr_db']:6.2f} dB\n")

ax1.set_title(f'Non-Coherent: Fin={Fin_arbitrary/1e6:.1f} MHz (spectral leakage!)')
ax2.set_title(f'Coherent: Fin={Fin_coherent/1e6:.3f} MHz (Bin {Fin_bin})')

plt.tight_layout()
fig_path = (output_dir / 'exp_s05_spectrum_leakage_demo.png').resolve()
print(f"[Save fig] -> [{fig_path}]\n")
plt.savefig(fig_path, dpi=150)
plt.close()