"""
Polar Phase Spectrum Analysis: Thermal Noise vs Harmonic Distortion

This example demonstrates polar spectrum visualization in two scenarios:

Left - Thermal Noise Only:
  - High noise (2 mVrms): Significant noise interference
  - Random phase distribution across all frequencies

Right - Harmonic Distortion:
  - HD2=-80dB, HD3=-50dB, k3 negative (stronger 3rd harmonic)
  - Fixed phase relationship between fundamental and harmonics
  - k3 polarity: Changes HD3 phase by 180°
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, amplitudes_to_snr, snr_to_nsd, analyze_spectrum_polar

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Common parameters
N = 2**13
Fs = 800e6
Fin_target = 80e6
Fin, Fin_bin = find_coherent_frequency(Fs, Fin_target, N)
t = np.arange(N) / Fs
A, DC = 0.49, 0.5

print(f"[Sinewave] Fs=[{Fs/1e6:.2f} MHz], Fin=[{Fin/1e6:.2f} MHz], Bin/N=[{Fin_bin}/{N}], A=[{A:.3f} Vpeak]")
print()

# Create 1x2 subplot grid with polar projection
fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': 'polar'})

# ============================================================================
# Left Plot: Thermal Noise Only (2 mVrms)
# ============================================================================
print("=" * 80)
print("LEFT: THERMAL NOISE (2 mVrms)")
print("=" * 80)

noise_rms = 2e-3
sig_ideal = A * np.sin(2*np.pi*Fin*t)
signal_noise = sig_ideal + DC + np.random.randn(N) * noise_rms

# Calculate theoretical values
snr_theory = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=noise_rms)
nsd_theory = snr_to_nsd(snr_theory, fs=Fs, osr=1)

plt.sca(axes[0])
result_noise = analyze_spectrum_polar(signal_noise, fs=Fs, fixed_radial_range=120)
axes[0].set_title('Thermal Noise: 2 mVrms (High)', pad=20, fontsize=12, fontweight='bold')

print(f"[2 mVrms] SNR={snr_theory:.1f}dB → Measured: ENoB={result_noise['enob']:.2f}b, SNR={result_noise['snr_db']:.2f}dB")

print()

# ============================================================================
# Right Plot: Harmonic Distortion (HD2=-80dB, HD3=-50dB, k3 negative)
# ============================================================================
print("=" * 80)
print("RIGHT: HARMONIC DISTORTION (HD2=-80dB, HD3=-50dB, k3<0)")
print("=" * 80)

base_noise = 500e-6
hd2_dB = -80
hd3_dB = -50

hd2_amp = 10**(hd2_dB/20)
hd3_amp = 10**(hd3_dB/20)
k2 = hd2_amp / (A / 2)
k3 = -hd3_amp / (A**2 / 4)  # Negative k3

signal_hd = sig_ideal + k2 * sig_ideal**2 + k3 * sig_ideal**3 + DC + np.random.randn(N) * base_noise

plt.sca(axes[1])
result_hd = analyze_spectrum_polar(signal_hd, fs=Fs, fixed_radial_range=120)
axes[1].set_title(f'HD2={hd2_dB}dB, HD3={hd3_dB}dB, k3<0\n(Thermal Noise: 500 uVrms)', pad=20, fontsize=12, fontweight='bold')

print(f"[HD2={hd2_dB}dB, HD3={hd3_dB}dB, k3<0] SNDR={result_hd['sndr_db']:.2f}dB, THD={result_hd['thd_db']:.2f}dB, HD2={result_hd['hd2_db']:.2f}dB, HD3={result_hd['hd3_db']:.2f}dB")

print()
print("=" * 80)

plt.tight_layout()
fig_path = output_dir / 'exp_s10_polar_noise_and_harmonics.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"\n[Save fig] -> [{fig_path}]")
plt.close()
