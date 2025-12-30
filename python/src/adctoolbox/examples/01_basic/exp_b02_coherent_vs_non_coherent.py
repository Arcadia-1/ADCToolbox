"""
Basic demo: Comparison of Coherent vs. Non-Coherent Sampling.

This script demonstrates the critical effect of spectral leakage by analyzing
the same signal with an arbitrary (non-coherent) frequency and a calculated
coherent frequency. It shows how coherent sampling reduces leakage.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, analyze_spectrum, amplitudes_to_snr, snr_to_nsd

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N_fft = 2**13
Fs = 100e6
A = 0.5
noise_rms = 50e-6

Fin_arbitrary = 10e6
Fin_coherent, Fin_bin = find_coherent_frequency(Fs, Fin_arbitrary, N_fft)

snr_ref = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=noise_rms)
nsd_ref = snr_to_nsd(snr_ref, fs=Fs, osr=1)
print(f"[Sinewave] Fs=[{Fs/1e6:.2f} MHz], Fin_arbitrary=[{Fin_arbitrary/1e6:.6f} MHz], Fin_coherent=[{Fin_coherent/1e6:.6f} MHz], Bin/N=[{Fin_bin}/{N_fft}], A=[{A:.3f} Vpeak]")
print(f"[Nonideal] Noise RMS=[{noise_rms*1e6:.2f} uVrms], Theoretical SNR=[{snr_ref:.2f} dB], Theoretical NSD=[{nsd_ref:.2f} dBFS/Hz]\n")

t = np.arange(N_fft) / Fs
signal_arbitrary = A * np.sin(2*np.pi*Fin_arbitrary*t)  + np.random.randn(N_fft) * noise_rms
signal_coherent = A * np.sin(2*np.pi*Fin_coherent*t) + np.random.randn(N_fft) * noise_rms

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

plt.sca(ax1)
result1 = analyze_spectrum(signal_arbitrary, fs=Fs)
print(f"[Non-coherent] ENoB=[{result1['enob']:5.2f} b], SNDR=[{result1['sndr_dbc']:6.2f} dB], SFDR=[{result1['sfdr_dbc']:6.2f} dB], SNR=[{result1['snr_dbc']:6.2f} dB], NSD=[{result1['nsd_dbfs_hz']:7.2f} dBFS/Hz]")

plt.sca(ax2)
result2 = analyze_spectrum(signal_coherent, fs=Fs)
print(f"[    Coherent] ENoB=[{result2['enob']:5.2f} b], SNDR=[{result2['sndr_dbc']:6.2f} dB], SFDR=[{result2['sfdr_dbc']:6.2f} dB], SNR=[{result2['snr_dbc']:6.2f} dB], NSD=[{result2['nsd_dbfs_hz']:7.2f} dBFS/Hz]")
ax1.set_title(f'Non-Coherent: Fin={Fin_arbitrary/1e6:.1f} MHz (spectral leakage!)')
ax2.set_title(f'Coherent: Fin={Fin_coherent/1e6:.3f} MHz (Bin {Fin_bin})')

plt.tight_layout()
fig_path = (output_dir / 'exp_b02_coherent_vs_non_coherent.png').resolve()
print(f"\n[Save fig] -> [{fig_path}]\n")
plt.savefig(fig_path, dpi=150)
plt.close()