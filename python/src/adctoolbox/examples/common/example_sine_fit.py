"""
Example: sine_fit - Sine Wave Fitting

This example demonstrates how to use sine_fit to extract sinewave parameters
from ADC data: frequency, amplitude, phase, and DC offset.

The sine_fit function uses least-squares optimization to fit a sinewave
model to the data: y = A*sin(2*pi*f*t + phi) + DC
"""

import numpy as np
import matplotlib.pyplot as plt
from adctoolbox.common import sine_fit, find_bin

# Create output directory
import os
output_dir = "../output"
os.makedirs(output_dir, exist_ok=True)

print("=" * 70)
print("Example: sine_fit - Sine Wave Parameter Extraction")
print("=" * 70)

#%% Example 1: Fit Clean Sinewave
print("\nExample 1: Fit Clean Sinewave")
print("-" * 70)

# Generate clean sinewave with known parameters
N = 2**10  # 1024 samples
Fs = 1.0
Fin_true = 0.1234  # True frequency
A_true = 0.49  # True amplitude
DC_true = 0.5  # True DC offset
phi_true = np.pi / 4  # True phase (45 degrees)

t = np.arange(N)
signal_clean = A_true * np.sin(2 * np.pi * Fin_true * t + phi_true) + DC_true

print(f"\nTrue Parameters:")
print(f"  Frequency: {Fin_true:.6f}")
print(f"  Amplitude: {A_true:.4f}")
print(f"  DC offset: {DC_true:.4f}")
print(f"  Phase:     {phi_true:.4f} rad ({np.degrees(phi_true):.2f}°)")

# Fit sinewave
signal_fit, freq_est, amp_est, dc_est, phi_est = sine_fit(signal_clean)

print(f"\nEstimated Parameters:")
print(f"  Frequency: {freq_est:.6f} (error: {abs(freq_est - Fin_true):.2e})")
print(f"  Amplitude: {amp_est:.4f} (error: {abs(amp_est - A_true):.2e})")
print(f"  DC offset: {dc_est:.4f} (error: {abs(dc_est - DC_true):.2e})")
print(f"  Phase:     {phi_est:.4f} rad ({np.degrees(phi_est):.2f}°, error: {abs(phi_est - phi_true):.2e} rad)")

# Calculate residual error
residual = signal_clean - signal_fit
print(f"\nFit Quality:")
print(f"  RMS error: {np.std(residual):.2e}")
print(f"  Max error: {np.max(np.abs(residual)):.2e}")

# Plot
fig1, axes = plt.subplots(2, 1, figsize=(12, 8))

# Time domain
axes[0].plot(t[:200], signal_clean[:200], 'b.-', label='Original', alpha=0.7)
axes[0].plot(t[:200], signal_fit[:200], 'r--', label='Fitted', linewidth=2)
axes[0].set_xlabel('Sample Index')
axes[0].set_ylabel('Amplitude')
axes[0].set_title('Sine Fit - Time Domain (First 200 samples)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Residual
axes[1].plot(t, residual, 'g-', alpha=0.7)
axes[1].set_xlabel('Sample Index')
axes[1].set_ylabel('Residual Error')
axes[1].set_title(f'Residual Error (RMS: {np.std(residual):.2e})')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'sine_fit_clean.png'), dpi=150)
plt.close()

#%% Example 2: Fit Noisy Sinewave
print("\n" + "=" * 70)
print("Example 2: Fit Noisy Sinewave")
print("-" * 70)

# Add noise to signal
noise_level = 0.01
signal_noisy = signal_clean + np.random.randn(N) * noise_level

print(f"\nNoise level (RMS): {noise_level}")

# Fit noisy signal
signal_fit_noisy, freq_noisy, amp_noisy, dc_noisy, phi_noisy = sine_fit(signal_noisy)

print(f"\nEstimated Parameters (with noise):")
print(f"  Frequency: {freq_noisy:.6f} (error: {abs(freq_noisy - Fin_true):.2e})")
print(f"  Amplitude: {amp_noisy:.4f} (error: {abs(amp_noisy - A_true):.2e})")
print(f"  DC offset: {dc_noisy:.4f} (error: {abs(dc_noisy - DC_true):.2e})")
print(f"  Phase:     {phi_noisy:.4f} rad (error: {abs(phi_noisy - phi_true):.2e} rad)")

residual_noisy = signal_noisy - signal_fit_noisy
print(f"\nFit Quality:")
print(f"  RMS error: {np.std(residual_noisy):.4f}")
print(f"  (Close to noise level: {noise_level})")

# Plot
fig2, axes = plt.subplots(2, 1, figsize=(12, 8))

# Time domain
axes[0].plot(t[:200], signal_noisy[:200], 'b.', label='Noisy Signal', alpha=0.5, markersize=3)
axes[0].plot(t[:200], signal_fit_noisy[:200], 'r-', label='Fitted', linewidth=2)
axes[0].set_xlabel('Sample Index')
axes[0].set_ylabel('Amplitude')
axes[0].set_title(f'Sine Fit with Noise (SNR ≈ {20*np.log10(A_true/noise_level):.1f} dB)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Residual
axes[1].plot(t, residual_noisy, 'g-', alpha=0.5)
axes[1].set_xlabel('Sample Index')
axes[1].set_ylabel('Residual Error')
axes[1].set_title(f'Residual Error (RMS: {np.std(residual_noisy):.4f})')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'sine_fit_noisy.png'), dpi=150)
plt.close()

#%% Example 3: Fit Signal with Harmonics
print("\n" + "=" * 70)
print("Example 3: Fit Signal with Harmonics (Fundamental Only)")
print("-" * 70)

# Add harmonics to signal
HD2_amp = 0.05  # 2nd harmonic
HD3_amp = 0.03  # 3rd harmonic

signal_harmonic = (signal_clean +
                   HD2_amp * np.sin(2 * 2 * np.pi * Fin_true * t) +
                   HD3_amp * np.sin(3 * 2 * np.pi * Fin_true * t))

print(f"\nSignal with harmonics:")
print(f"  HD2 amplitude: {HD2_amp:.4f}")
print(f"  HD3 amplitude: {HD3_amp:.4f}")

# Fit fundamental only
signal_fit_fund, freq_fund, amp_fund, dc_fund, phi_fund = sine_fit(signal_harmonic)

print(f"\nFitted Fundamental Parameters:")
print(f"  Frequency: {freq_fund:.6f}")
print(f"  Amplitude: {amp_fund:.4f}")
print(f"  DC offset: {dc_fund:.4f}")

residual_harmonic = signal_harmonic - signal_fit_fund
print(f"\nResidual (contains harmonics):")
print(f"  RMS: {np.std(residual_harmonic):.4f}")

# Plot
fig3, axes = plt.subplots(3, 1, figsize=(12, 10))

# Original signal
axes[0].plot(t[:200], signal_harmonic[:200], 'b-', label='Signal with Harmonics')
axes[0].set_xlabel('Sample Index')
axes[0].set_ylabel('Amplitude')
axes[0].set_title('Original Signal (Fundamental + Harmonics)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Fitted fundamental
axes[1].plot(t[:200], signal_fit_fund[:200], 'r-', label='Fitted Fundamental')
axes[1].set_xlabel('Sample Index')
axes[1].set_ylabel('Amplitude')
axes[1].set_title('Fitted Fundamental Component')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Residual (harmonics)
axes[2].plot(t[:200], residual_harmonic[:200], 'g-', label='Residual (Harmonics)')
axes[2].set_xlabel('Sample Index')
axes[2].set_ylabel('Amplitude')
axes[2].set_title('Residual = Harmonics + Noise')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'sine_fit_harmonics.png'), dpi=150)
plt.close()

#%% Example 4: Batch Fitting (Multiple Signals)
print("\n" + "=" * 70)
print("Example 4: Batch Fitting for Frequency Sweep")
print("-" * 70)

# Generate multiple signals with different frequencies
freq_list = np.linspace(0.05, 0.45, 10)
results = []

print(f"\nFitting {len(freq_list)} signals...")

for f_true in freq_list:
    sig = 0.49 * np.sin(2 * np.pi * f_true * t) + 0.5 + np.random.randn(N) * 0.001
    _, f_est, a_est, dc_est, phi_est = sine_fit(sig)
    results.append({
        'f_true': f_true,
        'f_est': f_est,
        'error': abs(f_est - f_true),
        'amp': a_est
    })

# Plot frequency accuracy
fig4, axes = plt.subplots(2, 1, figsize=(12, 8))

f_true_arr = [r['f_true'] for r in results]
f_error_arr = [r['error'] for r in results]
amp_arr = [r['amp'] for r in results]

# Frequency error
axes[0].plot(f_true_arr, f_error_arr, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('True Frequency (normalized)')
axes[0].set_ylabel('Frequency Error')
axes[0].set_title('Frequency Estimation Error vs Input Frequency')
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale('log')

# Amplitude estimation
axes[1].plot(f_true_arr, amp_arr, 'ro-', linewidth=2, markersize=8)
axes[1].axhline(y=0.49, color='k', linestyle='--', label='True Amplitude')
axes[1].set_xlabel('True Frequency (normalized)')
axes[1].set_ylabel('Estimated Amplitude')
axes[1].set_title('Amplitude Estimation vs Input Frequency')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'sine_fit_sweep.png'), dpi=150)
plt.close()

print(f"\nFrequency Estimation Accuracy:")
print(f"  Mean error:   {np.mean(f_error_arr):.2e}")
print(f"  Max error:    {np.max(f_error_arr):.2e}")
print(f"  Std dev:      {np.std(f_error_arr):.2e}")

#%% Summary
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(f"\nGenerated figures saved to: {output_dir}/")
print("  - sine_fit_clean.png      (Clean signal fit)")
print("  - sine_fit_noisy.png      (Noisy signal fit)")
print("  - sine_fit_harmonics.png  (Signal with harmonics)")
print("  - sine_fit_sweep.png      (Frequency sweep accuracy)")

print("\nKey Takeaways:")
print("  1. sine_fit extracts: frequency, amplitude, DC, phase")
print("  2. Works well even with noise (error ≈ noise level)")
print("  3. Fits fundamental only (residual contains harmonics)")
print("  4. Accuracy is consistent across frequency range")
print("  5. Use for: ADC characterization, signal analysis, error extraction")

print("\n" + "=" * 70)
print("Example completed successfully!")
print("=" * 70)
