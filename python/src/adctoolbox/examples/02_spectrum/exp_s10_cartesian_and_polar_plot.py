"""
Spectrum Analysis: Cartesian Plot vs Polar Plot

This example demonstrates static nonlinearity (harmonic distortion) visualization:

Left - Cartesian Plot:
  - Standard FFT spectrum plot in dBFS
  - Shows signal, harmonics, and noise floor

Right - Polar Plot:
  - Polar visualization of the same signal
  - HD2=-80dB, HD3=-50dB, k3 negative (stronger 3rd harmonic)
  - Fixed phase relationship between fundamental and harmonics
  - k3 polarity: Changes HD3 phase by 180°
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, analyze_spectrum, analyze_spectrum_polar

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Common parameters
N = 2**13
Fs = 800e6
Fin_target = 60e6
Fin, Fin_bin = find_coherent_frequency(Fs, Fin_target, N)
t = np.arange(N) / Fs
A, DC = 0.49, 0.5

print(f"[Sinewave] Fs=[{Fs/1e6:.2f} MHz], Fin=[{Fin/1e6:.2f} MHz], Bin/N=[{Fin_bin}/{N}], A=[{A:.3f} Vpeak]")
print()

# ============================================================================
# Static Nonlinearity: Harmonic Distortion (HD2=-80dB, HD3=-50dB, k3 negative)
# ============================================================================
print("=" * 80)
print("STATIC NONLINEARITY: HARMONIC DISTORTION (HD2=-80dB, HD3=-50dB, k3<0)")
print("=" * 80)

base_noise = 500e-6
hd2_dB = -80
hd3_dB = -50

sig_ideal = A * np.sin(2*np.pi*Fin*t)
hd2_amp = 10**(hd2_dB/20)
hd3_amp = 10**(hd3_dB/20)
k2 = hd2_amp / (A / 2)
k3 = -hd3_amp / (A**2 / 4)  # Negative k3

signal_hd = sig_ideal + k2 * sig_ideal**2 + k3 * sig_ideal**3 + DC + np.random.randn(N) * base_noise

# Create 1x2 subplot grid (left: normal, right: polar)
fig = plt.figure(figsize=(14, 6))
ax_left = fig.add_subplot(1, 2, 1)
ax_right = fig.add_subplot(1, 2, 2, projection='polar')

# Left Plot: Cartesian Plot
plt.sca(ax_left)
result_normal = analyze_spectrum(signal_hd, fs=Fs)
ax_left.set_title(f'Cartesian Plot\nHD2={hd2_dB}dB, HD3={hd3_dB}dB, k3<0', fontsize=12, fontweight='bold')

print(f"[Cartesian Plot] SNDR={result_normal['sndr_dbc']:.2f}dB, THD={result_normal['thd_dbc']:.2f}dB, HD2={result_normal['harmonics_dbc'][0]:.2f}dB, HD3={result_normal['harmonics_dbc'][1]:.2f}dB")

# Right Plot: Polar Plot
plt.sca(ax_right)
result_polar = analyze_spectrum_polar(signal_hd, fs=Fs, fixed_radial_range=120)
ax_right.set_title(f'Polar Plot\nHD2={hd2_dB}dB, HD3={hd3_dB}dB, k3<0', pad=20, fontsize=12, fontweight='bold')

print(f"[Polar Plot] SNDR={result_polar['sndr_dbc']:.2f}dB, THD={result_polar['thd_dbc']:.2f}dB, HD2={result_polar['harmonics_dbc'][0]:.2f}dB, HD3={result_polar['harmonics_dbc'][1]:.2f}dB")

print()

plt.tight_layout()
fig_path = output_dir / 'exp_s10_cartesian_and_polar_plot.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"\n[Save fig] -> [{fig_path}]")
plt.close()
