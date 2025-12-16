"""Test core fit_sine_harmonics function - LS fitting kernel

This example demonstrates the fit_sine_harmonics function, which is the
fundamental least-squares fitting kernel used by all analysis modes.

Tests:
1. Single harmonic fitting (order=1)
2. Multi-harmonic fitting (order=5)
3. Accuracy of phase and amplitude extraction
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, amplitudes_to_snr, snr_to_nsd
from adctoolbox.aout import fit_sine_harmonics

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Setup parameters
N = 2**13
Fs = 800e6
Fin_target = 10e6
Fin, Fin_bin = find_coherent_frequency(Fs, Fin_target, N)
normalized_freq = Fin / Fs
t = np.arange(N) / Fs
A = 0.49
noise_rms = 100e-6  # 100 uV RMS thermal noise

snr_fundamental = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=noise_rms)
nsd_fundamental = snr_to_nsd(snr_fundamental, fs=Fs, osr=1)
print(f"[Sinewave] Fs={Fs/1e6:.1f} MHz, Fin={Fin/1e6:.6f} MHz, A={A:.3f} V, N={N}")
print(f"[Nonideal] Noise RMS=[{noise_rms*1e6:.2f} uVrms], Theoretical SNR=[{snr_fundamental:.2f} dB], Theoretical NSD=[{nsd_fundamental:.2f} dBFS/Hz]")

hd2_dB, hd3_dB = -80, -70
k2 = 10**(hd2_dB/20) / (A / 2)
k3 = 10**(hd3_dB/20) / (A**2 / 4)

sig_ideal = A * np.sin(2 * np.pi * Fin * t)
sig_with_noise = sig_ideal + np.random.randn(N) * noise_rms
sig_with_harmonics = sig_ideal + k2 * sig_ideal**2 + k3 * sig_ideal**3 + np.random.randn(N) * noise_rms

print(f"[Harmonic Specification] HD2={hd2_dB}dB, HD3={hd3_dB}dB, k2={k2:.6f}, k3={k3:.6f}\n")

print("\n" + "="*70)
print("Test 1: Single Harmonic Fitting (order=1)")
print("="*70)

W1, sig_fit1, A1, phase1 = fit_sine_harmonics(
    sig_ideal,
    freq=normalized_freq,
    order=1,
    include_dc=True
)

mag1 = np.sqrt(W1[1]**2 + W1[2]**2)
print(f"DC offset: {W1[0]:.6e}")
print(f"Fundamental magnitude: {mag1:.6e}")
print(f"Phase: {phase1 * 180/np.pi:.2f}°")
print(f"Fit error (RMS): {np.sqrt(np.mean((sig_ideal - sig_fit1)**2)):.6e}")

print("\n" + "="*70)
print("Test 2: Multi-Harmonic Fitting (order=5)")
print("="*70)

W5, sig_fit5, A5, phase5 = fit_sine_harmonics(
    sig_with_harmonics,
    freq=normalized_freq,
    order=5,
    include_dc=True
)

print(f"Coefficients: {len(W5)} (DC + 2*5 harmonics)")
print("Harmonic magnitudes:")
for h in range(1, 4):
    mag = np.sqrt(W5[2*h-1]**2 + W5[2*h]**2)
    print(f"  H{h}: {mag:.6e}")
print(f"Fit error (RMS): {np.sqrt(np.mean((sig_with_harmonics - sig_fit5)**2)):.6e}")

print("\n" + "="*70)
print("Test 3: Noisy Signal Fitting")
print("="*70)

W_noisy, sig_fit_noisy, A_noisy, phase_noisy = fit_sine_harmonics(
    sig_with_noise,
    freq=normalized_freq,
    order=1,
    include_dc=True
)

residual = sig_with_noise - sig_fit_noisy
mag_noisy = np.sqrt(W_noisy[1]**2 + W_noisy[2]**2)
print(f"Signal noise RMS: {noise_rms:.6e}")
print(f"DC offset: {W_noisy[0]:.6e}")
print(f"Fundamental magnitude: {mag_noisy:.6e}")
print(f"Residual error RMS: {np.sqrt(np.mean(residual**2)):.6e}")

# ============================================================
# Visualization
# ============================================================
print("\n[Generating plots...]")

residual1 = sig_ideal - sig_fit1
residual5 = sig_with_harmonics - sig_fit5
sample_range = slice(0, 1000)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('LS Fit Kernel Tests', fontsize=16, fontweight='bold')

# Plot 1: Single harmonic fitting
axes[0, 0].plot(t[sample_range], sig_ideal[sample_range], 'b-', label='Original', linewidth=1.5)
axes[0, 0].plot(t[sample_range], sig_fit1[sample_range], 'r--', label='Fitted (order=1)', linewidth=1.5)
axes[0, 0].set_title('Test 1: Single Harmonic Fitting')

# Plot 2: Multi-harmonic fitting
axes[0, 1].plot(t[sample_range], sig_with_harmonics[sample_range], 'b-', label='Original (with H2, H3)', linewidth=1.5)
axes[0, 1].plot(t[sample_range], sig_fit5[sample_range], 'r--', label='Fitted (order=5)', linewidth=1.5)
axes[0, 1].set_title('Test 2: Multi-Harmonic Fitting')

# Plot 3: Fitting residuals
axes[1, 0].plot(t[sample_range], residual1[sample_range], 'b-', label='Single harmonic residual', linewidth=1)
axes[1, 0].plot(t[sample_range], residual5[sample_range], 'r-', label='Multi-harmonic residual', linewidth=1)
axes[1, 0].set_title('Fitting Residuals')

# Plot 4: Noisy signal fitting
axes[1, 1].plot(t[sample_range], sig_with_noise[sample_range], 'b-', label='Noisy signal', linewidth=1, alpha=0.7)
axes[1, 1].plot(t[sample_range], sig_fit_noisy[sample_range], 'r--', label='Fitted', linewidth=2)
axes[1, 1].set_title('Test 3: Noisy Signal')

# Common formatting for all subplots
for ax in axes.flat:
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude' if ax is not axes[1, 0] else 'Residual')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = output_dir / 'exp_a51_fit_sinewave_components.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"[Save fig] -> [{fig_path}]")
plt.close(fig)

print("\n[Complete!]")
