"""
Polar phase spectrum analysis: Static Nonlinearity vs Memory Effect.

This example demonstrates polar spectrum visualization in two scenarios:

Row 1 - Static Nonlinearity (3 cases):
  - HD3=-66dB, k3 positive
  - HD3=-66dB, k3 negative
  - HD2=-80dB + HD3=-66dB combined

Row 2 - Memory Effect (3 frequencies):
  - MSB-dependent memory effect at different input frequencies
  - Shows how spur phases vary with Fin/Fs ratio
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
A, DC = 0.49, 0.5
base_noise = 50e-6

snr_ref = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=base_noise)
nsd_ref = snr_to_nsd(snr_ref, fs=Fs, osr=1)
print(f"[Sinewave] Fs=[{Fs/1e6:.0f} MHz], N=[{N}], A=[{A:.3f} Vpeak]")
print(f"[Nonideal] Noise RMS=[{base_noise*1e6:.2f} uVrms], Theoretical SNR=[{snr_ref:.2f} dB], Theoretical NSD=[{nsd_ref:.2f} dBFS/Hz]\n")

# Create 2x3 subplot grid with polar projection
fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw={'projection': 'polar'})

# ============================================================================
# Row 1: Static Nonlinearity (3 cases)
# ============================================================================
print("=" * 80)
print("ROW 1: STATIC NONLINEARITY")
print("=" * 80)

Fin_static = 80e6
Fin, Fin_bin = find_coherent_frequency(Fs, Fin_static, N)
t = np.arange(N) / Fs
sig_ideal = A * np.sin(2*np.pi*Fin*t)

print(f"[Fin={Fin/1e6:.2f} MHz], Bin/N=[{Fin_bin}/{N}]\n")

# Compute nonlinearity coefficients
hd2_dB = -80
hd3_dB = -66
hd2_amp = 10**(hd2_dB/20)
hd3_amp = 10**(hd3_dB/20)
k2 = hd2_amp / (A / 2)
k3 = hd3_amp / (A**2 / 4)

# Case 1: HD3 only, k3 positive
signal_1 = sig_ideal + k3 * sig_ideal**3 + DC + np.random.randn(N) * base_noise
plt.sca(axes[0, 0])
result_1 = analyze_spectrum_polar(signal_1, fs=Fs, fixed_radial_range=120)
axes[0, 0].set_title(f'HD3={hd3_dB}dB, k3>0\n(Thermal Noise: 500 uVrms)', pad=20, fontsize=12, fontweight='bold')
print(f"[HD3={hd3_dB}dB, k3>0] SNDR={result_1['sndr_dbc']:.2f}dB, THD={result_1['thd_dbc']:.2f}dB, HD3={result_1['harmonics_dbc'][1]:.2f}dB")

# Case 2: HD3 only, k3 negative
signal_2 = sig_ideal - k3 * sig_ideal**3 + DC + np.random.randn(N) * base_noise
plt.sca(axes[0, 1])
result_2 = analyze_spectrum_polar(signal_2, fs=Fs, fixed_radial_range=120)
axes[0, 1].set_title(f'HD3={hd3_dB}dB, k3<0\n(Thermal Noise: 500 uVrms)', pad=20, fontsize=12, fontweight='bold')
print(f"[HD3={hd3_dB}dB, k3<0] SNDR={result_2['sndr_dbc']:.2f}dB, THD={result_2['thd_dbc']:.2f}dB, HD3={result_2['harmonics_dbc'][1]:.2f}dB")

# Case 3: HD2 + HD3 combined
signal_3 = sig_ideal + k2 * sig_ideal**2 - k3 * sig_ideal**3 + DC + np.random.randn(N) * base_noise
plt.sca(axes[0, 2])
result_3 = analyze_spectrum_polar(signal_3, fs=Fs, fixed_radial_range=120)
axes[0, 2].set_title(f'HD2={hd2_dB}dB + HD3={hd3_dB}dB\n(Thermal Noise: 500 uVrms)', pad=20, fontsize=12, fontweight='bold')
print(f"[HD2+HD3] SNDR={result_3['sndr_dbc']:.2f}dB, THD={result_3['thd_dbc']:.2f}dB, HD2={result_3['harmonics_dbc'][0]:.2f}dB, HD3={result_3['harmonics_dbc'][1]:.2f}dB")

print()

# ============================================================================
# Row 2: Memory Effect (3 frequencies)
# ============================================================================
print("=" * 80)
print("ROW 2: MEMORY EFFECT")
print("=" * 80)

# 3 different input frequencies
Fin_targets = [40e6, 80e6, 160e6]
memory_effect_strength = 0.02

for col_idx, Fin_target in enumerate(Fin_targets):
    Fin, J = find_coherent_frequency(Fs, Fin_target, N)

    # Generate clean signal for this frequency
    t_ext = np.arange(N+1) / Fs
    sig_clean_ext = A * np.sin(2*np.pi*Fin*t_ext) + DC + np.random.randn(N+1) * base_noise
    msb_ext = np.floor(sig_clean_ext * 2**4) / 2**4
    lsb_ext = np.floor((sig_clean_ext - msb_ext) * 2**18) / 2**18
    msb_shifted = msb_ext[:-1]
    msb = msb_ext[1:]
    lsb = lsb_ext[1:]

    # Expected phase delay per sample
    phase_delay_deg = 360 * Fin / Fs

    # Generate memory effect signal
    signal_me = msb + lsb + memory_effect_strength * msb_shifted

    # Analyze spectrum with polar phase visualization
    plt.sca(axes[1, col_idx])
    result = analyze_spectrum_polar(
        signal_me,
        fs=Fs,
        harmonic=5,
        win_type='boxcar',
        fixed_radial_range=120
    )

    # Calculate theoretical harmonic phases
    # HD2: follows simple 2φ relationship
    # HD3: follows 180° - 3φ (phase inverted due to memory effect mechanism)
    hd2_phase_theory = (2 * phase_delay_deg) % 360
    hd3_phase_theory = (180 - 3 * phase_delay_deg) % 360

    # Set title with theoretical phase information
    title = f'Fin={Fin/1e6:.0f}MHz (φ={phase_delay_deg:.1f}°), ME={memory_effect_strength}\nHD2∠{hd2_phase_theory:.1f}°, HD3∠{hd3_phase_theory:.1f}°'
    axes[1, col_idx].set_title(title, pad=20, fontsize=12, fontweight='bold')

    print(f"[Fin={Fin/1e6:5.0f}MHz, ME={memory_effect_strength}] sndr={result['sndr_dbc']:5.2f}dB, snr={result['snr_dbc']:5.2f}dB, thd={result['thd_dbc']:6.2f}dB")

print()

plt.tight_layout()

fig_path = output_dir / 'exp_s11_polar_memory_effect.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n[Save fig] -> [{fig_path}]")
