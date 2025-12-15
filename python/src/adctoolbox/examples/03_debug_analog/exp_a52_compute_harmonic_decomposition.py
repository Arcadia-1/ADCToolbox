"""Test compute_harmonic_decomposition - signal decomposition

This example demonstrates harmonic decomposition using the new modular approach.
Shows how to compute fundamental, harmonic, and residual components using
the compute_harmonic_decomposition function.

Tests:
1. Signal with static nonlinearity (harmonics)
2. Harmonic decomposition accuracy with varying orders
3. Comparison of different harmonic orders (3 vs 5 vs 10)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, amplitudes_to_snr, snr_to_nsd
from adctoolbox.aout import compute_harmonic_decomposition

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

print(f"[Harmonic Decomposition Test] Fs={Fs/1e6:.1f} MHz, Fin={Fin/1e6:.6f} MHz, A={A:.3f} V, N={N}\n")

# Generate test signals
sig_ideal = A * np.sin(2 * np.pi * Fin * t)

# ADC Specification: HD2=-80dB, HD3=-70dB (12-bit standard)
hd2_dB, hd3_dB = -80, -70
hd2_amp = 10**(hd2_dB/20)
hd3_amp = 10**(hd3_dB/20)
k2 = hd2_amp / (A / 2)
k3 = hd3_amp / (A**2 / 4)

# Case 1: Thermal noise
print("="*70)
print("Case 1: Thermal Noise Only")
print("="*70)

noise_rms = 100e-6
sig_noise = sig_ideal + np.random.randn(N) * noise_rms

snr_noise = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=noise_rms)
nsd_noise = snr_to_nsd(snr_noise, fs=Fs, osr=1)
print(f"[Noise Signal] Noise RMS=[{noise_rms*1e6:.2f} uVrms], Theoretical SNR=[{snr_noise:.2f} dB], Theoretical NSD=[{nsd_noise:.2f} dBFS/Hz]\n")

decomp_noise = compute_harmonic_decomposition(sig_noise, normalized_freq=normalized_freq, order=10)
print(f"Fundamental power: {np.mean(decomp_noise['fundamental_signal']**2):.6e}")
print(f"Harmonic power: {np.mean(decomp_noise['harmonic_error']**2):.6e}")
print(f"Other error power: {np.mean(decomp_noise['other_error']**2):.6e}")
print(f"Total power: {np.mean(sig_noise**2):.6e}")

# Case 2: Static nonlinearity
print("\n" + "="*70)
print(f"Case 2: Static Nonlinearity (HD2={hd2_dB}dB, HD3={hd3_dB}dB)")
print("="*70)

base_noise = 10e-6
sig_nonlin = sig_ideal + k2 * sig_ideal**2 + k3 * sig_ideal**3 + np.random.randn(N) * base_noise

snr_nonlin = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=base_noise)
nsd_nonlin = snr_to_nsd(snr_nonlin, fs=Fs, osr=1)
print(f"[Nonlin Signal] Noise RMS=[{base_noise*1e6:.2f} uVrms], HD2={hd2_dB}dB, HD3={hd3_dB}dB, Theoretical SNR=[{snr_nonlin:.2f} dB], Theoretical NSD=[{nsd_nonlin:.2f} dBFS/Hz]\n")

decomp_nonlin = compute_harmonic_decomposition(sig_nonlin, normalized_freq=normalized_freq, order=10)
print(f"Fundamental power: {np.mean(decomp_nonlin['fundamental_signal']**2):.6e}")
print(f"Harmonic power: {np.mean(decomp_nonlin['harmonic_error']**2):.6e}")
print(f"Other error power: {np.mean(decomp_nonlin['other_error']**2):.6e}")
print(f"Total power: {np.mean(sig_nonlin**2):.6e}")

# Case 3: Different order comparisons
print("\n" + "="*70)
print("Case 3: Order Comparison (order=3 vs 5 vs 10)")
print("="*70)

for order in [3, 5, 10]:
    decomp = compute_harmonic_decomposition(sig_nonlin, normalized_freq=normalized_freq, order=order)
    harmonic_pwr = np.mean(decomp['harmonic_error']**2)
    other_pwr = np.mean(decomp['other_error']**2)
    print(f"Order={order:2d}: Harmonic power={harmonic_pwr:.6e}, Other power={other_pwr:.6e}")

# ============================================================
# Visualization
# ============================================================
print("\n[Generating plots...]")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Harmonic Decomposition Tests', fontsize=16, fontweight='bold')

sample_range = slice(0, 2000)

# Plot 1: Thermal noise - time domain
ax = axes[0, 0]
ax.plot(t[sample_range], sig_noise[sample_range], 'b-', label='Input (noise)', linewidth=0.8, alpha=0.7)
ax.plot(t[sample_range], decomp_noise['fundamental_signal'][sample_range], 'r-', label='Fundamental', linewidth=1.5)
ax.plot(t[sample_range], decomp_noise['other_error'][sample_range], 'g--', label='Residual', linewidth=1)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.set_title(f'Case 1: Thermal Noise ({noise_rms*1e6:.0f}μV RMS)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Static nonlinearity - time domain
ax = axes[0, 1]
ax.plot(t[sample_range], sig_nonlin[sample_range], 'b-', label='Input (nonlinear)', linewidth=0.8, alpha=0.7)
ax.plot(t[sample_range], decomp_nonlin['fundamental_signal'][sample_range], 'r-', label='Fundamental', linewidth=1.5)
ax.plot(t[sample_range], decomp_nonlin['harmonic_error'][sample_range], 'orange', label='Harmonics', linewidth=1.2)
ax.plot(t[sample_range], decomp_nonlin['other_error'][sample_range], 'g--', label='Residual', linewidth=1)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.set_title(f'Case 2: Static Nonlinearity (k2={k2:.3f}, k3={k3:.3f})')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Power distribution - noise case
ax = axes[1, 0]
categories = ['Fundamental', 'Harmonic', 'Other']
powers = [
    np.mean(decomp_noise['fundamental_signal']**2),
    np.mean(decomp_noise['harmonic_error']**2),
    np.mean(decomp_noise['other_error']**2)
]
bars = ax.bar(categories, powers, color=['red', 'orange', 'green'], alpha=0.7)
ax.set_ylabel('Power (V²)')
ax.set_title('Power Distribution - Thermal Noise')
ax.set_yscale('log')
for bar, pwr in zip(bars, powers):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{pwr:.2e}', ha='center', va='bottom', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Power distribution - nonlinear case
ax = axes[1, 1]
powers_nonlin = [
    np.mean(decomp_nonlin['fundamental_signal']**2),
    np.mean(decomp_nonlin['harmonic_error']**2),
    np.mean(decomp_nonlin['other_error']**2)
]
bars = ax.bar(categories, powers_nonlin, color=['red', 'orange', 'green'], alpha=0.7)
ax.set_ylabel('Power (V²)')
ax.set_title('Power Distribution - Static Nonlinearity')
ax.set_yscale('log')
for bar, pwr in zip(bars, powers_nonlin):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{pwr:.2e}', ha='center', va='bottom', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig_path = output_dir / 'exp_a52_harmonic_decomposition.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"[Save fig] -> [{fig_path}]")
plt.close(fig)

# Additional figure: Order comparison
fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle('Harmonic Decomposition - Order Comparison', fontsize=14, fontweight='bold')

orders = [3, 5, 7, 10]
harmonic_powers = []
other_powers = []

for order in orders:
    decomp = compute_harmonic_decomposition(sig_nonlin, normalized_freq=normalized_freq, order=order)
    harmonic_powers.append(np.mean(decomp['harmonic_error']**2))
    other_powers.append(np.mean(decomp['other_error']**2))

x = np.arange(len(orders))
width = 0.35

bars1 = ax.bar(x - width/2, harmonic_powers, width, label='Harmonic Power', alpha=0.8)
bars2 = ax.bar(x + width/2, other_powers, width, label='Other Power', alpha=0.8)

ax.set_xlabel('Decomposition Order')
ax.set_ylabel('Power (V²)')
ax.set_title('Effect of Harmonic Order on Decomposition')
ax.set_xticks(x)
ax.set_xticklabels([f'order={o}' for o in orders])
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig_path = output_dir / 'exp_a52_harmonic_order_comparison.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"[Save fig] -> [{fig_path}]")
plt.close(fig)

print("\n[Complete!]")
