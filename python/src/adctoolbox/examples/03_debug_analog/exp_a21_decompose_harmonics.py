"""Harmonic decomposition: thermal noise vs static nonlinearity"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, analyze_harmonic_decomposition, amplitudes_to_snr, snr_to_nsd

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N = 2**13
Fs = 800e6
Fin_target = 10e6
Fin, Fin_bin = find_coherent_frequency(Fs, Fin_target, N)
t = np.arange(N) / Fs
A = 0.49

sig_ideal = A * np.sin(2 * np.pi * Fin * np.arange(N) / Fs)

# Case 1: Thermal noise only
noise_rms = 500e-6
signal_noise = sig_ideal + np.random.randn(N) * noise_rms

snr_ref = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=noise_rms)
nsd_ref = snr_to_nsd(snr_ref, fs=Fs, osr=1)
print(f"[Harmonic Decomposition] Fs={Fs/1e6:.1f} MHz, Fin={Fin/1e6:.6f} MHz, Bin={Fin_bin}, N_fft={N}")
print(f"[Thermal Noise] Noise RMS=[{noise_rms*1e6:.2f} uVrms], Theoretical SNR=[{snr_ref:.2f} dB], Theoretical NSD=[{nsd_ref:.2f} dBFS/Hz]\n")

# Case 2: Static nonlinearity (k2 and k3) + base noise
k2 = 0.001
k3 = 0.005
base_noise_rms = 50e-6
signal_nonlin = sig_ideal + k2 * sig_ideal**2 + k3 * sig_ideal**3 + np.random.randn(N) * base_noise_rms

snr_nonlin = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=base_noise_rms)
nsd_nonlin = snr_to_nsd(snr_nonlin, fs=Fs, osr=1)
print(f"[Static Nonlin] Noise RMS=[{base_noise_rms*1e6:.2f} uVrms], k2={k2:.4f}, k3={k3:.4f}, Theoretical SNR=[{snr_nonlin:.2f} dB], Theoretical NSD=[{nsd_nonlin:.2f} dBFS/Hz]\n")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle('Harmonic Decomposition', fontsize=16, fontweight='bold')

plt.sca(ax1)
analyze_harmonic_decomposition(signal_noise, normalized_freq=Fin/Fs, order=10, show_plot=True)
ax1.set_title(f'Case 1: Thermal Noise ({noise_rms*1e6:.0f}uV RMS)', fontsize=14, fontweight='bold')

plt.sca(ax2)
analyze_harmonic_decomposition(signal_nonlin, normalized_freq=Fin/Fs, order=10, show_plot=True)
ax2.set_title(f'Case 2: Static Nonlinearity (k2={k2:.3f}, k3={k3:.3f})', fontsize=14, fontweight='bold')

plt.tight_layout()
fig_path = output_dir / 'exp_a21_decompose_harmonics.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\n[Save fig] -> [{fig_path}]\n")
plt.close(fig)