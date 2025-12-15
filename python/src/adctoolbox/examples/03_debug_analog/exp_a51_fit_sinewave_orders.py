"""Multi-Order Sinewave Fitting - Orders 1 to 5 with Residues

This example demonstrates fitting sinewaves at different orders (1st through 5th)
and shows how the fitting improves with higher orders by examining residuals.

Visualization: 2 rows x 5 columns (top row = fitted signals, bottom row = residuals)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, amplitudes_to_snr, snr_to_nsd
from adctoolbox.aout import fit_sinewave_components

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

print(f"[Sinewave Fitting - Multi-Order] Fs={Fs/1e6:.1f} MHz, Fin={Fin/1e6:.6f} MHz, A={A:.3f} V, N={N}")

# Generate ideal signal
sig_ideal = A * np.sin(2 * np.pi * Fin * t)

# Add harmonics to make multi-order fitting meaningful
hd2_dB, hd3_dB = -80, -70
hd2_amp = 10**(hd2_dB/20)
hd3_amp = 10**(hd3_dB/20)
k2 = hd2_amp / (A / 2)
k3 = hd3_amp / (A**2 / 4)

sig_with_harmonics = sig_ideal + k2 * sig_ideal**2 + k3 * sig_ideal**3

# Add noise
noise_rms = 100e-6
sig_noisy = sig_with_harmonics + np.random.randn(N) * noise_rms

snr_signal = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=noise_rms)
nsd_signal = snr_to_nsd(snr_signal, fs=Fs, osr=1)
print(f"[Signal] Noise RMS=[{noise_rms*1e6:.2f} uVrms], Theoretical SNR=[{snr_signal:.2f} dB], Theoretical NSD=[{nsd_signal:.2f} dBFS/Hz]")
print(f"[Harmonics] HD2={hd2_dB}dB, HD3={hd3_dB}dB, k2={k2:.6f}, k3={k3:.6f}\n")

# Fit different orders
print("="*70)
print("Multi-Order Fitting Analysis (Orders 1-5)")
print("="*70)

orders = [1, 2, 3, 4, 5]
fits = {}
residuals = {}
errors_rms = {}

for order in orders:
    W, sig_fit, A_matrix, phase = fit_sinewave_components(
        sig_noisy,
        freq=normalized_freq,
        order=order,
        include_dc=True
    )

    fits[order] = sig_fit
    residuals[order] = sig_noisy - sig_fit
    errors_rms[order] = np.sqrt(np.mean(residuals[order]**2))

    # Print summary
    fundamental_mag = np.sqrt(W[1]**2 + W[2]**2) if len(W) > 2 else 0
    print(f"Order {order}: RMS residual = {errors_rms[order]:.6e} V, Fundamental = {fundamental_mag:.6e} V")

print()

# ============================================================
# Visualization: 2x5 grid (top row = fitted, bottom row = residuals)
# ============================================================
print("[Generating 2x5 residue plots...]")

sample_range = slice(0, 2000)
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('Multi-Order Sinewave Fitting with Residuals (Orders 1-5)', fontsize=16, fontweight='bold')

for idx, order in enumerate(orders):
    # Top row: Fitted signals
    ax_fit = axes[0, idx]
    ax_fit.plot(t[sample_range], sig_noisy[sample_range], 'b-', label='Noisy input', linewidth=0.8, alpha=0.6)
    ax_fit.plot(t[sample_range], fits[order][sample_range], 'r-', label='Fitted', linewidth=2)
    ax_fit.set_title(f'Order {order} - Fitted Signal')
    ax_fit.set_xlabel('Time (s)')
    ax_fit.set_ylabel('Amplitude (V)')
    ax_fit.legend(fontsize=9)
    ax_fit.grid(True, alpha=0.3)

    # Bottom row: Residuals
    ax_res = axes[1, idx]
    ax_res.plot(t[sample_range], residuals[order][sample_range], 'g-', linewidth=0.8)
    ax_res.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
    ax_res.set_title(f'Order {order} - Residual (RMS={errors_rms[order]:.2e})')
    ax_res.set_xlabel('Time (s)')
    ax_res.set_ylabel('Residual (V)')
    ax_res.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = output_dir / 'exp_a51_fit_sinewave_orders_2x5.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"[Save fig] -> [{fig_path}]")
plt.close(fig)

# ============================================================
# Additional figure: Residual RMS vs Order
# ============================================================
print("[Generating order comparison plot...]")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Residual RMS vs Order
order_list = list(orders)
error_list = [errors_rms[o] for o in order_list]

ax1.plot(order_list, error_list, 'b-o', linewidth=2, markersize=8)
ax1.axhline(y=noise_rms, color='r', linestyle='--', linewidth=2, label=f'Input noise RMS = {noise_rms:.2e}')
ax1.set_xlabel('Fitting Order')
ax1.set_ylabel('Residual RMS (V)')
ax1.set_title('Residual RMS vs Fitting Order')
ax1.set_xticks(order_list)
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_yscale('log')

# Plot 2: Error reduction
error_reduction = [100 * (error_list[0] - error_list[i]) / error_list[0] for i in range(len(error_list))]
ax2.bar(order_list, error_reduction, color='skyblue', edgecolor='navy', alpha=0.7)
ax2.set_xlabel('Fitting Order')
ax2.set_ylabel('Error Reduction (%)')
ax2.set_title('Cumulative Error Reduction from Order 1')
ax2.set_xticks(order_list)
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (order, reduction) in enumerate(zip(order_list, error_reduction)):
    ax2.text(order, reduction + 1, f'{reduction:.1f}%', ha='center', fontsize=10)

fig.suptitle('Order Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
fig_path = output_dir / 'exp_a51_fit_sinewave_order_analysis.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"[Save fig] -> [{fig_path}]")
plt.close(fig)

print("\n[Complete!]")
