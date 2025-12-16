"""Code-based error analysis: Thermal noise vs 3rd order nonlinearity

This example demonstrates the analyze_error_by_code wrapper function for detecting
static nonlinearity errors (INL/DNL) and code-dependent noise.

Compares two cases:
1. Ideal ADC with thermal noise (200 uVrms)
2. ADC with 3rd order nonlinearity (k3=0.01)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, amplitudes_to_snr, snr_to_nsd
from adctoolbox.aout import analyze_error_by_code
from adctoolbox.aout import rearrange_error_by_code
from adctoolbox.aout.plot_error_binned_code import plot_error_binned_code

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Setup parameters
N = 2**14
Fs = 800e6
Fin_target = 10e6
Fin, Fin_bin = find_coherent_frequency(Fs, Fin_target, N)
normalized_freq = Fin / Fs
t = np.arange(N) / Fs
A = 0.49
num_bits = 10
LSB = 1.0 / (2**num_bits)

print(f"[Code Error Analysis] Fs={Fs/1e6:.1f} MHz, Fin={Fin/1e6:.6f} MHz, num_bits={num_bits}")
print(f"LSB = {LSB:.6f}\n")

# ============================================================
# Demonstration: High-Level Wrapper Function
# ============================================================
print("="*70)
print("Demonstration: analyze_error_by_code Wrapper Function")
print("="*70)
print("\nThis shows the user-friendly analyze_error_by_code wrapper with auto-plotting.\n")

A_demo = 0.49
base_noise_demo = 200e-6  # 200 uVrms
sig_demo = A_demo * np.sin(2 * np.pi * Fin * t) + np.random.randn(N) * base_noise_demo

print("[High-Level Interface - analyze_error_by_code]")
print("Returns: (emean_by_code, erms_by_code, code_bins)\n")

emean_demo, erms_demo, codes_demo = analyze_error_by_code(
    sig_demo,
    normalized_freq=normalized_freq,
    num_bits=num_bits,
    show_plot=False
)

print(f"  Number of codes: {len(codes_demo)}")
print(f"  Max mean error: {np.nanmax(np.abs(emean_demo)):.6e} LSB")
print(f"  Max RMS error: {np.nanmax(erms_demo):.6e} LSB")
print(f"  Mean RMS across codes: {np.nanmean(erms_demo[~np.isnan(erms_demo)]):.6e} LSB")

print("\n" + "="*70 + "\n")

# ============================================================
# Case 1: Thermal Noise Only (200 uVrms)
# ============================================================
print("="*70)
print("Case 1: Thermal Noise (200 uVrms)")
print("="*70)

noise_rms = 200e-6
sig_noise = A * np.sin(2 * np.pi * Fin * t) + np.random.randn(N) * noise_rms

snr_noise = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=noise_rms)
nsd_noise = snr_to_nsd(snr_noise, fs=Fs, osr=1)
print(f"[Signal] A={A:.3f} V, Noise RMS=[{noise_rms*1e6:.2f} uVrms], SNR=[{snr_noise:.2f} dB], NSD=[{nsd_noise:.2f} dBFS/Hz]\n")

emean_noise, erms_noise, codes_noise = analyze_error_by_code(
    sig_noise,
    normalized_freq=normalized_freq,
    num_bits=num_bits,
    show_plot=False
)

print(f"Number of codes used: {np.sum(~np.isnan(emean_noise))}")
print(f"Max mean error: {np.nanmax(np.abs(emean_noise)):.6e}")
print(f"Max RMS error: {np.nanmax(erms_noise):.6e}")
print(f"Mean RMS across codes: {np.nanmean(erms_noise[~np.isnan(erms_noise)]):.6e}\n")

# ============================================================
# Case 2: 3rd Order Nonlinearity (k3=0.01)
# ============================================================
print("="*70)
print("Case 2: 3rd Order Nonlinearity (k3=0.01)")
print("="*70)

k3 = 0.01
base_noise = 50e-6
sig_nonlin = A * np.sin(2 * np.pi * Fin * t) + k3 * (A * np.sin(2 * np.pi * Fin * t))**3 + np.random.randn(N) * base_noise

snr_nonlin = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=base_noise)
nsd_nonlin = snr_to_nsd(snr_nonlin, fs=Fs, osr=1)
print(f"[Signal] A={A:.3f} V, k3={k3:.4f}, Noise RMS=[{base_noise*1e6:.2f} uVrms], SNR=[{snr_nonlin:.2f} dB], NSD=[{nsd_nonlin:.2f} dBFS/Hz]\n")

emean_nonlin, erms_nonlin, codes_nonlin = analyze_error_by_code(
    sig_nonlin,
    normalized_freq=normalized_freq,
    num_bits=num_bits,
    show_plot=False
)

print(f"Number of codes used: {np.sum(~np.isnan(emean_nonlin))}")
print(f"Max mean error: {np.nanmax(np.abs(emean_nonlin)):.6e}")
print(f"Max RMS error: {np.nanmax(erms_nonlin):.6e}")
print(f"Mean RMS across codes: {np.nanmean(erms_nonlin[~np.isnan(erms_nonlin)]):.6e}\n")

# ============================================================
# Visualization
# ============================================================
print("\n" + "="*70)
print("VISUALIZATION USING rearrange_error_by_code WITH PLOTTING")
print("="*70 + "\n")

# Compute error decomposition for both cases
print("[Computing Case 1: Thermal Noise (200 uVrms)]")
results_noise = rearrange_error_by_code(sig_noise, normalized_freq=normalized_freq, num_bits=num_bits)

print("[Computing Case 2: 3rd Order Nonlinearity (k3=0.01)]")
results_nonlin = rearrange_error_by_code(sig_nonlin, normalized_freq=normalized_freq, num_bits=num_bits)

# Create figure with 2 columns (one for each case)
# Each column will be subdivided into 2 subplots by plot_error_binned_code
fig = plt.figure(figsize=(18, 8))
fig.suptitle('Code Error Analysis - Thermal Noise vs 3rd Order Nonlinearity', fontsize=16, fontweight='bold')

# Case 1: Thermal Noise (left column)
print("[Plotting Case 1: Thermal Noise (200 uVrms)]")
ax1 = plt.subplot(1, 2, 1)
plot_error_binned_code(results_noise, ax=ax1)

# Case 2: 3rd Order Nonlinearity (right column)
print("[Plotting Case 2: 3rd Order Nonlinearity (k3=0.01)]")
ax2 = plt.subplot(1, 2, 2)
plot_error_binned_code(results_nonlin, ax=ax2)

plt.tight_layout()
fig_path = output_dir / 'exp_a10_analyze_error_by_code.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\n[Save fig] -> [{fig_path}]\n")
plt.close(fig)

# ============================================================
# Comparison Summary
# ============================================================
print("="*70)
print("SUMMARY: Case Comparison")
print("="*70)

print("\nCase 1 (Thermal Noise - 200 uVrms):")
print(f"  Max mean error: {np.nanmax(np.abs(emean_noise)):.6e} LSB")
print(f"  Max RMS error: {np.nanmax(erms_noise):.6e} LSB")
print(f"  Mean RMS: {np.nanmean(erms_noise[~np.isnan(erms_noise)]):.6e} LSB")

print("\nCase 2 (3rd Order Nonlinearity - k3=0.01):")
print(f"  Max mean error: {np.nanmax(np.abs(emean_nonlin)):.6e} LSB")
print(f"  Max RMS error: {np.nanmax(erms_nonlin):.6e} LSB")
print(f"  Mean RMS: {np.nanmean(erms_nonlin[~np.isnan(erms_nonlin)]):.6e} LSB")

print("\nKey Observation:")
print(f"  Nonlinearity error is {np.nanmax(np.abs(emean_nonlin))/np.nanmax(np.abs(emean_noise)):.1f}x larger than thermal noise")

print("\n[Complete!]")
