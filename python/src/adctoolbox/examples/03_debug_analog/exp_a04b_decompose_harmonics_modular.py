"""Harmonic decomposition: thermal noise vs static nonlinearity (Modular Version)

This example demonstrates the new modular architecture for LMS harmonic decomposition:
1. calculate_lms_decomposition() - Pure calculation
2. plot_decomposition_time() - Time domain visualization
3. plot_decomposition_polar() - Polar plot visualization

Compares two cases:
- Case 1: Thermal noise only (noise appears in "other errors")
- Case 2: Static nonlinearity (harmonics appear as distinct components)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency
from adctoolbox.aout import (
    calculate_lms_decomposition,
    plot_decomposition_time,
    plot_decomposition_polar,
    analyze_decomposition_time,
    analyze_decomposition_polar,
)

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Setup parameters
N = 2**13
Fs = 800e6
Fin_target = 10e6
Fin, Fin_bin = find_coherent_frequency(Fs, Fin_target, N)
t = np.arange(N) / Fs
A = 0.49

# Ideal signal
sig_ideal = A * np.sin(2 * np.pi * Fin * t)
print(f"[LMS Harmonic Decomposition - Modular] Fs={Fs/1e6:.1f} MHz, Fin={Fin/1e6:.6f} MHz, Bin={Fin_bin}, N_fft={N}")

# Case 1: Thermal noise only
noise_rms = 500e-6
signal_noise = sig_ideal + np.random.randn(N) * noise_rms

# Case 2: Static nonlinearity (k2 and k3) + base noise
k2 = 0.001
k3 = 0.005
base_noise_rms = 50e-6
signal_nonlin = sig_ideal + k2 * sig_ideal**2 + k3 * sig_ideal**3 + np.random.randn(N) * base_noise_rms

print("\n[Modular Structure Demonstration]")
print("  Step 1: calculate_lms_decomposition() - Pure calculation")
print("  Step 2: plot_decomposition_time() or plot_decomposition_polar() - Pure visualization")
print("  Or use wrappers: analyze_decomposition_time() or analyze_decomposition_polar()")

# ============================================================
# Part 1: Time-domain comparison using modular approach
# ============================================================
print("\n[Part 1: Time Domain Comparison]")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle('Harmonic Decomposition - Time Domain (Modular)', fontsize=16, fontweight='bold')

# Case 1: Thermal noise
print("  Case 1: Thermal noise")
decomp_noise = calculate_lms_decomposition(signal_noise, harmonic=10)

# Prepare plot data for Case 1
signal_mean = np.mean(signal_noise)
plot_data_noise = {
    'signal': signal_noise - signal_mean,
    'fundamental_signal': decomp_noise['fundamental_signal'] * (np.max(signal_noise) - np.min(signal_noise)),
    'harmonic_signal': decomp_noise['harmonic_signal'] * (np.max(signal_noise) - np.min(signal_noise)),
    'residual': decomp_noise['residual'] * (np.max(signal_noise) - np.min(signal_noise)),
    'fundamental_freq': decomp_noise['fundamental_freq'],
    'title': f'Case 1: Thermal Noise ({noise_rms*1e6:.0f}uV RMS)',
}

plot_decomposition_time(plot_data_noise, ax=ax1)
print(f"    Noise floor: {decomp_noise['noise_dB']:.1f} dB")

# Case 2: Static nonlinearity
print("  Case 2: Static nonlinearity")
decomp_nonlin = calculate_lms_decomposition(signal_nonlin, harmonic=10)

# Prepare plot data for Case 2
signal_mean = np.mean(signal_nonlin)
plot_data_nonlin = {
    'signal': signal_nonlin - signal_mean,
    'fundamental_signal': decomp_nonlin['fundamental_signal'] * (np.max(signal_nonlin) - np.min(signal_nonlin)),
    'harmonic_signal': decomp_nonlin['harmonic_signal'] * (np.max(signal_nonlin) - np.min(signal_nonlin)),
    'residual': decomp_nonlin['residual'] * (np.max(signal_nonlin) - np.min(signal_nonlin)),
    'fundamental_freq': decomp_nonlin['fundamental_freq'],
    'title': f'Case 2: Static Nonlinearity (k2={k2:.3f}, k3={k3:.3f})',
}

plot_decomposition_time(plot_data_nonlin, ax=ax2)
print(f"    Noise floor: {decomp_nonlin['noise_dB']:.1f} dB")

plt.tight_layout()
fig_path_time = output_dir / 'exp_a04b_decompose_time_modular.png'
plt.savefig(fig_path_time, dpi=150, bbox_inches='tight')
print(f"\n[Save fig] -> [{fig_path_time}]")
plt.close(fig)

# ============================================================
# Part 2: Polar comparison using modular approach
# ============================================================
print("\n[Part 2: Polar Comparison (LMS Mode)]")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), subplot_kw=dict(projection='polar'))
fig.suptitle('Harmonic Decomposition - Polar (LMS Mode, Modular)', fontsize=16, fontweight='bold')

# Case 1: Thermal noise - polar plot
print("  Case 1: Thermal noise - polar")
plot_data_noise_polar = {
    'harm_mag': decomp_noise['harm_mag'],
    'harm_phase': decomp_noise['harm_phase'],
    'harm_dB': decomp_noise['harm_dB'],
    'noise_dB': decomp_noise['noise_dB'],
    'harmonic': 10,
    'title': f'Case 1: Thermal Noise\n({noise_rms*1e6:.0f}uV RMS)',
}

plot_decomposition_polar(plot_data_noise_polar, ax=ax1)

# Case 2: Static nonlinearity - polar plot
print("  Case 2: Static nonlinearity - polar")
plot_data_nonlin_polar = {
    'harm_mag': decomp_nonlin['harm_mag'],
    'harm_phase': decomp_nonlin['harm_phase'],
    'harm_dB': decomp_nonlin['harm_dB'],
    'noise_dB': decomp_nonlin['noise_dB'],
    'harmonic': 10,
    'title': f'Case 2: Static Nonlinearity\n(k2={k2:.3f}, k3={k3:.3f})',
}

plot_decomposition_polar(plot_data_nonlin_polar, ax=ax2)

# Print harmonic magnitudes
print("\n  Harmonic Magnitudes (dB):")
print("  Harmonic | Noise Case | Nonlin Case")
print("  ---------|------------|------------")
for i in range(5):
    print(f"     {i+1:2d}    |  {decomp_noise['harm_dB'][i]:7.1f}   |  {decomp_nonlin['harm_dB'][i]:7.1f}")

plt.tight_layout()
fig_path_polar = output_dir / 'exp_a04b_decompose_polar_modular.png'
plt.savefig(fig_path_polar, dpi=150, bbox_inches='tight')
print(f"\n[Save fig] -> [{fig_path_polar}]")
plt.close(fig)

# ============================================================
# Part 3: Demonstrate high-level wrappers
# ============================================================
print("\n[Part 3: High-level Wrappers Demo]")
print("  Using analyze_decomposition_time() and analyze_decomposition_polar()")

# Time-domain wrapper
decomp_result, plot_data = analyze_decomposition_time(
    signal_nonlin,
    harmonic=10,
    fs=Fs,
    title=f'Static Nonlinearity (k2={k2:.3f}, k3={k3:.3f}) - Wrapper Demo',
    save_path=output_dir / 'exp_a04b_wrapper_time.png',
    show_plot=False
)
print(f"  Time wrapper saved -> [{output_dir / 'exp_a04b_wrapper_time.png'}]")

# Polar wrapper
decomp_result, plot_data = analyze_decomposition_polar(
    signal_nonlin,
    harmonic=10,
    fs=Fs,
    title=f'Static Nonlinearity (k2={k2:.3f}, k3={k3:.3f}) - Wrapper Demo',
    save_path=output_dir / 'exp_a04b_wrapper_polar.png',
    show_plot=False
)
print(f"  Polar wrapper saved -> [{output_dir / 'exp_a04b_wrapper_polar.png'}]")

print("\n[Modular Architecture Summary]")
print("  Calculation Engines:")
print("    - calculate_lms_decomposition() (NEW)")
print("    - calculate_coherent_spectrum() (from previous work)")
print("    - calculate_spectrum_metrics() (existing)")
print("\n  Visualization Functions:")
print("    - plot_decomposition_time() (NEW)")
print("    - plot_decomposition_polar() (NEW)")
print("    - plot_polar_phase() (from previous work)")
print("    - plot_spectrum() (existing)")
print("\n  High-level Wrappers (User-facing):")
print("    - analyze_decomposition_time() (NEW)")
print("    - analyze_decomposition_polar() (NEW)")
print("    - analyze_spectrum_polar() (NEW)")
print("    - analyze_spectrum() (existing)")
print("\n[Complete!]")
