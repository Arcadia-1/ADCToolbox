"""
Polar phase spectrum analysis: visualize FFT bins as magnitude-phase vectors in polar coordinates.
Demonstrates harmonic distortion with positive and negative k3 coefficients showing how
HD2 and HD3 phases vary with nonlinearity polarity.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, amplitudes_to_snr, snr_to_nsd, analyze_spectrum_polar

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N = 2**13
Fs = 800e6
Fin_target = 80e6
Fin, Fin_bin = find_coherent_frequency(Fs, Fin_target, N)
t = np.arange(N) / Fs
A, DC = 0.49, 0.5
base_noise = 50e-6

# Harmonic distortion levels
hd2_dB, hd3_dB = -80, -66
hd2_amp = 10**(hd2_dB/20)  # Harmonic amplitude / Fundamental amplitude
hd3_amp = 10**(hd3_dB/20)

# Compute nonlinearity coefficients to achieve target HD levels
# For y = x + k2*x² + k3*x³:
# HD2: k2 * A^2 / 2 = hd2_amp * A  →  k2 = hd2_amp / (A/2)
# HD3: k3 * A^3 / 4 = hd3_amp * A  →  k3 = hd3_amp / (A²/4)
k2 = hd2_amp / (A / 2)
k3_pos = hd3_amp / (A**2 / 4)
k3_neg = -k3_pos

# Signal 1: Harmonic distortion with positive k3
sig_ideal = A * np.sin(2*np.pi*Fin*t)
signal_hd_pos_k3 = (sig_ideal + k2 * sig_ideal**2 + k3_pos * sig_ideal**3 + DC + np.random.randn(N) * base_noise)

# Signal 2: Harmonic distortion with negative k3
signal_hd_neg_k3 = (sig_ideal + k2 * sig_ideal**2 + k3_neg * sig_ideal**3 + DC + np.random.randn(N) * base_noise)

signals = [signal_hd_pos_k3, signal_hd_neg_k3]
description = [f'k2 = {k2:.6f}, k3 = {k3_pos:9.6f}',
          f'k2 = {k2:.6f}, k3 = {k3_neg:9.6f}']

snr_ref = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=base_noise)
nsd_ref = snr_to_nsd(snr_ref, fs=Fs, osr=1)
print(f"[Sinewave] Fs=[{Fs/1e6:.2f} MHz], Fin=[{Fin/1e6:.2f} MHz], Bin/N=[{Fin_bin}/{N}], A=[{A:.3f} Vpeak]")
print(f"[Base Noise] RMS=[{base_noise*1e6:.2f} uVrms], Theoretical SNR=[{snr_ref:.2f} dB], Theoretical NSD=[{nsd_ref:.2f} dBFS/Hz]")
print(f"[Nonlinearity] HD2=[{hd2_dB} dB], HD3=[{hd3_dB} dB]\n")

fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'projection': 'polar'})

for i, (signal, title) in enumerate(zip(signals, description)):
    plt.sca(axes[i])
    result = analyze_spectrum_polar(signal, fs=Fs, fixed_radial_range=120)
    axes[i].set_title(f'{title}', pad=20, fontsize=14, fontweight='bold')

    print(f"[{title:16s}] sndr=[{result['sndr_db']:.2f} dB], snr=[{result['snr_db']:.2f} dB], "
          f"thd=[{result['thd_db']:.2f} dB],hd2=[{result['hd2_db']:.2f} dB], hd3=[{result['hd3_db']:.2f} dB]")

plt.tight_layout()
fig_path = output_dir / 'exp_s21_polar_harmonic.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"\n[Save fig] -> [{fig_path}]")
plt.close()
