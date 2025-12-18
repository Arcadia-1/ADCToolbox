"""
Polar Phase Spectrum Analysis: Thermal Noise vs Harmonic Distortion

This example demonstrates polar spectrum visualization in two scenarios:

Row 1 - Thermal Noise Only (3 noise levels):
  - Low noise (50 uVrms): Clear signal, random noise phase
  - Medium noise (500 uVrms): Increased noise floor
  - High noise (2 mVrms): Significant noise interference

Row 2 - Harmonic Distortion (3 cases):
  - HD2=-80dB, HD3=-66dB, k3 positive
  - HD2=-80dB, HD3=-66dB, k3 negative
  - HD2=-80dB, HD3=-50dB, k3 negative (stronger 3rd harmonic)

Key Observations:
- Thermal noise: Random phase distribution across all frequencies
- Harmonic distortion: Fixed phase relationship between fundamental and harmonics
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

# Create 2x3 subplot grid with polar projection
fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw={'projection': 'polar'})

# ============================================================================
# Row 1: Thermal Noise Only (3 levels)
# ============================================================================
print("=" * 80)
print("ROW 1: THERMAL NOISE ONLY")
print("=" * 80)

noise_levels = [50e-6, 500e-6, 2e-3]
noise_labels = ['50 uVrms (Low)', '500 uVrms (Medium)', '2 mVrms (High)']

for i, (noise_rms, label) in enumerate(zip(noise_levels, noise_labels)):
    # Generate ideal sinewave with thermal noise only
    sig_ideal = A * np.sin(2*np.pi*Fin*t)
    signal = sig_ideal + DC + np.random.randn(N) * noise_rms

    # Calculate theoretical values
    snr_theory = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=noise_rms)
    nsd_theory = snr_to_nsd(snr_theory, fs=Fs, osr=1)

    plt.sca(axes[0, i])
    result = analyze_spectrum_polar(signal, fs=Fs, fixed_radial_range=120)
    axes[0, i].set_title(f'Thermal Noise: {label}', pad=20, fontsize=12, fontweight='bold')

    print(f"[{label}] SNR={snr_theory:.1f}dB → Measured: ENoB={result['enob']:.2f}b, SNR={result['snr_db']:.2f}dB")

print()

# ============================================================================
# Row 2: Harmonic Distortion (3 cases)
# ============================================================================
print("=" * 80)
print("ROW 2: HARMONIC DISTORTION")
print("=" * 80)

base_noise = 500e-6

# Case 1: HD2=-80dB, HD3=-66dB, k3 positive
hd2_dB_1, hd3_dB_1 = -80, -66
hd2_amp_1 = 10**(hd2_dB_1/20)
hd3_amp_1 = 10**(hd3_dB_1/20)
k2_1 = hd2_amp_1 / (A / 2)
k3_1 = hd3_amp_1 / (A**2 / 4)

sig_ideal = A * np.sin(2*np.pi*Fin*t)
signal_1 = sig_ideal + k2_1 * sig_ideal**2 + k3_1 * sig_ideal**3 + DC + np.random.randn(N) * base_noise

plt.sca(axes[1, 0])
result_1 = analyze_spectrum_polar(signal_1, fs=Fs, fixed_radial_range=120)
axes[1, 0].set_title(f'HD2={hd2_dB_1}dB, HD3={hd3_dB_1}dB, k3>0\n(Thermal Noise: 500 uVrms)', pad=20, fontsize=12, fontweight='bold')
print(f"[HD2={hd2_dB_1}dB, HD3={hd3_dB_1}dB, k3>0] SNDR={result_1['sndr_db']:.2f}dB, THD={result_1['thd_db']:.2f}dB, HD2={result_1['hd2_db']:.2f}dB, HD3={result_1['hd3_db']:.2f}dB")

# Case 2: HD2=-80dB, HD3=-66dB, k3 negative
k3_2 = -k3_1
signal_2 = sig_ideal + k2_1 * sig_ideal**2 + k3_2 * sig_ideal**3 + DC + np.random.randn(N) * base_noise

plt.sca(axes[1, 1])
result_2 = analyze_spectrum_polar(signal_2, fs=Fs, fixed_radial_range=120)
axes[1, 1].set_title(f'HD2={hd2_dB_1}dB, HD3={hd3_dB_1}dB, k3<0\n(Thermal Noise: 500 uVrms)', pad=20, fontsize=12, fontweight='bold')
print(f"[HD2={hd2_dB_1}dB, HD3={hd3_dB_1}dB, k3<0] SNDR={result_2['sndr_db']:.2f}dB, THD={result_2['thd_db']:.2f}dB, HD2={result_2['hd2_db']:.2f}dB, HD3={result_2['hd3_db']:.2f}dB")

# Case 3: HD2=-80dB, HD3=-50dB (stronger), k3 negative
hd3_dB_3 = -50
hd3_amp_3 = 10**(hd3_dB_3/20)
k3_3 = -hd3_amp_3 / (A**2 / 4)

signal_3 = sig_ideal + k2_1 * sig_ideal**2 + k3_3 * sig_ideal**3 + DC + np.random.randn(N) * base_noise

plt.sca(axes[1, 2])
result_3 = analyze_spectrum_polar(signal_3, fs=Fs, fixed_radial_range=120)
axes[1, 2].set_title(f'HD2={hd2_dB_1}dB, HD3={hd3_dB_3}dB, k3<0\n(Thermal Noise: 500 uVrms)', pad=20, fontsize=12, fontweight='bold')
print(f"[HD2={hd2_dB_1}dB, HD3={hd3_dB_3}dB, k3<0] SNDR={result_3['sndr_db']:.2f}dB, THD={result_3['thd_db']:.2f}dB, HD2={result_3['hd2_db']:.2f}dB, HD3={result_3['hd3_db']:.2f}dB")

print()
print("=" * 80)

plt.tight_layout()
fig_path = output_dir / 'exp_s10_polar_noise_and_harmonics.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"\n[Save fig] -> [{fig_path}]")
plt.close()
