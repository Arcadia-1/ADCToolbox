"""Harmonic decomposition: thermal noise vs static nonlinearity

This example demonstrates the consolidated ADCToolbox analysis functions:
1. analyze_harmonic_decomposition - Decompose signal into harmonics
2. analyze_error_by_phase - Separate AM/PM error components
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, analyze_harmonic_decomposition, amplitudes_to_snr, snr_to_nsd
from adctoolbox.aout import analyze_error_by_phase

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

# ============================================================
# Analysis 1: Harmonic Decomposition
# ============================================================
print("="*70)
print("ANALYSIS 1: HARMONIC DECOMPOSITION")
print("="*70)

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

# ============================================================
# Analysis 2: AM/PM Error Decomposition (Binned Mode)
# ============================================================
print("="*70)
print("ANALYSIS 2: AM/PM ERROR DECOMPOSITION (Binned Mode)")
print("="*70)

print("\nCase 1: Thermal Noise")
am1_bin, pm1_bin, bl1_bin, erms1, emean1 = analyze_error_by_phase(
    signal_noise, normalized_freq=Fin/Fs, mode="binned", bin_count=100, show_plot=True
)
plt.suptitle('Case 1: AM/PM Decomposition - Thermal Noise (Binned Mode)', fontweight='bold')
fig_path = output_dir / 'exp_a21_am_pm_noise_binned.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"[Save fig] -> [{fig_path}]")
print(f"  AM param: {am1_bin:.6e}")
print(f"  PM param: {pm1_bin:.6e} rad")
print(f"  Baseline: {bl1_bin:.6e}\n")
plt.close()

print("Case 2: Static Nonlinearity")
am2_bin, pm2_bin, bl2_bin, erms2, emean2 = analyze_error_by_phase(
    signal_nonlin, normalized_freq=Fin/Fs, mode="binned", bin_count=100, show_plot=True
)
plt.suptitle('Case 2: AM/PM Decomposition - Static Nonlinearity (Binned Mode)', fontweight='bold')
fig_path = output_dir / 'exp_a21_am_pm_nonlin_binned.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"[Save fig] -> [{fig_path}]")
print(f"  AM param: {am2_bin:.6e}")
print(f"  PM param: {pm2_bin:.6e} rad")
print(f"  Baseline: {bl2_bin:.6e}\n")
plt.close()

# ============================================================
# Analysis 3: AM/PM Error Decomposition (Raw Mode - High Precision)
# ============================================================
print("="*70)
print("ANALYSIS 3: AM/PM ERROR DECOMPOSITION (Raw Mode - High Precision)")
print("="*70)

print("\nCase 1: Thermal Noise")
am1_raw, pm1_raw, bl1_raw, rms1 = analyze_error_by_phase(
    signal_noise, normalized_freq=Fin/Fs, mode="raw", show_plot=True
)
plt.suptitle('Case 1: AM/PM Decomposition - Thermal Noise (Raw Mode)', fontweight='bold')
fig_path = output_dir / 'exp_a21_am_pm_noise_raw.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"[Save fig] -> [{fig_path}]")
print(f"  AM param: {am1_raw:.6e}")
print(f"  PM param: {pm1_raw:.6e} rad")
print(f"  Baseline: {bl1_raw:.6e}")
print(f"  Error RMS: {rms1:.6e}\n")
plt.close()

print("Case 2: Static Nonlinearity")
am2_raw, pm2_raw, bl2_raw, rms2 = analyze_error_by_phase(
    signal_nonlin, normalized_freq=Fin/Fs, mode="raw", show_plot=True
)
plt.suptitle('Case 2: AM/PM Decomposition - Static Nonlinearity (Raw Mode)', fontweight='bold')
fig_path = output_dir / 'exp_a21_am_pm_nonlin_raw.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"[Save fig] -> [{fig_path}]")
print(f"  AM param: {am2_raw:.6e}")
print(f"  PM param: {pm2_raw:.6e} rad")
print(f"  Baseline: {bl2_raw:.6e}")
print(f"  Error RMS: {rms2:.6e}\n")
plt.close()

# ============================================================
# Summary Comparison
# ============================================================
print("="*70)
print("SUMMARY: Binned vs Raw Mode Comparison")
print("="*70)
print("\nCase 1 (Thermal Noise):")
print(f"  Binned Mode - AM: {am1_bin:.6e}, PM: {pm1_bin:.6e}")
print(f"  Raw Mode   - AM: {am1_raw:.6e}, PM: {pm1_raw:.6e}")
print(f"  AM diff: {abs(am1_bin - am1_raw):.6e}, PM diff: {abs(pm1_bin - pm1_raw):.6e}")

print("\nCase 2 (Static Nonlinearity):")
print(f"  Binned Mode - AM: {am2_bin:.6e}, PM: {pm2_bin:.6e}")
print(f"  Raw Mode   - AM: {am2_raw:.6e}, PM: {pm2_raw:.6e}")
print(f"  AM diff: {abs(am2_bin - am2_raw):.6e}, PM diff: {abs(pm2_bin - pm2_raw):.6e}")

print("\n[Complete!]")