"""
Example: fg_cal_sine - Foreground Calibration using Sinewave

This example demonstrates how to use fg_cal_sine to calibrate ADC bit weights
using a sinewave test signal. The tool estimates per-bit weights and DC offset
to correct for capacitor mismatch and other nonlinearities.

Key Features:
- Estimates calibrated bit weights
- Calculates DC offset
- Provides calibrated output signal
- Compares before/after performance
"""

import numpy as np
import matplotlib.pyplot as plt
from adctoolbox.dout import fg_cal_sine
from adctoolbox.aout import spec_plot
from adctoolbox.common import find_bin
from adctoolbox.examples.data import get_example_data_path

# Create output directory
import os
output_dir = "../output"
os.makedirs(output_dir, exist_ok=True)

print("=" * 70)
print("Example: fg_cal_sine - Foreground Calibration using Sinewave")
print("=" * 70)

#%% Example 0: Calibrate Real Digital Output Data
print("\nExample 0: Calibrate Real Digital Output Data")
print("-" * 70)

# Load example bits data (included with package)
data_file = get_example_data_path('dout_SAR_12b_weight_1.csv')
bits_real = np.loadtxt(data_file, delimiter=',', dtype=int)

print(f"\nLoaded example data:")
print(f"  File: dout_SAR_12b_weight_1.csv (12-bit SAR ADC)")
print(f"  Samples: {bits_real.shape[0]}")
print(f"  Bits: {bits_real.shape[1]}")
print(f"\nFirst 3 samples:")
for i in range(min(3, bits_real.shape[0])):
    print(f"  {i}: {bits_real[i]}")

# Run calibration on real data
weight_real, offset_real, aout_cal_real, aout_ideal_real, err_real, freq_real = fg_cal_sine(
    bits_real,
    freq=0,  # Auto-detect frequency
    order=5
)

print(f"\nCalibration Results:")
print(f"  Detected frequency: {freq_real:.6f}")
print(f"  DC offset: {offset_real:.6f}")
print(f"  Calibration error RMS: {np.std(err_real):.6f}")
print(f"\nEstimated weights:")
for i, w in enumerate(weight_real):
    print(f"  Bit {bits_real.shape[1]-1-i:2d}: {w:.6f}")

# Analyze performance
fig_real, axes = plt.subplots(1, 2, figsize=(16, 6))

# Uncalibrated (using ideal binary weights)
ideal_weights = 2.0 ** np.arange(bits_real.shape[1] - 1, -1, -1)
aout_uncal_real = np.dot(bits_real, ideal_weights)

plt.sca(axes[0])
enob_uncal, sndr_uncal, _, _, _, _, _, _ = spec_plot(aout_uncal_real, label=True, harmonic=5)
axes[0].set_title(f'Before Calibration\nENoB: {enob_uncal:.2f} bits')

plt.sca(axes[1])
enob_cal, sndr_cal, _, _, _, _, _, _ = spec_plot(aout_cal_real, label=True, harmonic=5)
axes[1].set_title(f'After Calibration\nENoB: {enob_cal:.2f} bits')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fg_cal_real_data.png'), dpi=150)
plt.close()

print(f"\nPerformance Improvement:")
print(f"  ENoB:  {enob_uncal:.2f} → {enob_cal:.2f} bits (+{enob_cal - enob_uncal:.2f})")
print(f"  SNDR:  {sndr_uncal:.2f} → {sndr_cal:.2f} dB (+{sndr_cal - sndr_uncal:.2f})")

#%% Additional Educational Examples with Synthetic Data
print("\n" + "=" * 70)
print("Additional Educational Examples with Synthetic Data")
print("=" * 70)

#%% Generate Test Signal with Bit Weight Errors
print("\nStep 1: Generate ADC digital codes with bit weight errors")
print("-" * 70)

# Parameters
N = 2**12  # Number of samples
resolution = 10  # ADC resolution (bits)
Fin_norm = 0.0789  # Normalized frequency

# Generate ideal sinewave
J = find_bin(1, Fin_norm, N)
Fin = J / N
t = np.arange(N)
A = 0.49
DC = 0.5

ideal_signal = A * np.sin(2 * np.pi * Fin * t) + DC

print(f"  Samples: {N}")
print(f"  Resolution: {resolution} bits")
print(f"  Frequency bin: {J}")
print(f"  Normalized frequency: {Fin:.6f}")

# Ideal (binary) weights
ideal_weights = 2.0 ** np.arange(resolution - 1, -1, -1)
print(f"\nIdeal binary weights:")
print(f"  {ideal_weights}")

# Create non-ideal weights (capacitor mismatch)
# Simulate random weight errors (±2% for each bit)
np.random.seed(42)
weight_errors = 1 + np.random.randn(resolution) * 0.02
actual_weights = ideal_weights * weight_errors

print(f"\nActual (non-ideal) weights with mismatch:")
print(f"  {actual_weights}")
print(f"\nWeight errors (%):")
for i in range(resolution):
    error_pct = (weight_errors[i] - 1) * 100
    print(f"  Bit {resolution-1-i:2d}: {error_pct:+.2f}%")

# Quantize signal to digital codes using actual weights
# First, quantize to ideal binary codes
digital_codes = np.floor((ideal_signal) * (2**resolution))
digital_codes = np.clip(digital_codes, 0, 2**resolution - 1).astype(int)

# Convert to binary matrix (MSB to LSB)
bits = np.zeros((N, resolution), dtype=int)
for i in range(resolution):
    bits[:, i] = (digital_codes >> (resolution - 1 - i)) & 1

# Reconstruct signal using non-ideal weights
uncalibrated_signal = np.dot(bits, actual_weights)

print(f"\nDigital code range: {digital_codes.min()} to {digital_codes.max()}")
print(f"Binary matrix shape: {bits.shape}")

#%% Analyze Uncalibrated Performance
print("\nStep 2: Analyze uncalibrated performance")
print("-" * 70)

fig1 = plt.figure(figsize=(12, 8))
enob_uncal, sndr_uncal, sfdr_uncal, snr_uncal, thd_uncal, _, _, _ = spec_plot(
    uncalibrated_signal,
    label=True,
    harmonic=5,
    osr=1,
    win_type='hamming'
)
plt.title('Uncalibrated ADC Output - Spectrum')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fg_cal_sine_uncalibrated.png'), dpi=150)
plt.close()

print(f"\nUncalibrated Performance:")
print(f"  ENoB:  {enob_uncal:.2f} bits")
print(f"  SNDR:  {sndr_uncal:.2f} dB")
print(f"  SFDR:  {sfdr_uncal:.2f} dB")

#%% Run Foreground Calibration
print("\nStep 3: Run foreground calibration")
print("-" * 70)

# Method 1: Auto-detect frequency (freq=0)
print("\nMethod 1: Auto-detect frequency")
weight_cal, offset_cal, post_cal, ideal, err, freq_cal = fg_cal_sine(
    bits,
    freq=0,      # Auto-detect frequency
    order=5      # Polynomial order for calibration
)

print(f"\nCalibration Results:")
print(f"  Detected frequency: {freq_cal:.6f}")
print(f"  DC offset: {offset_cal:.6f}")
print(f"  Error RMS: {np.std(err):.6f}")

print(f"\nCalibrated weights:")
print(f"  {weight_cal}")

print(f"\nWeight correction factors:")
for i in range(resolution):
    correction = weight_cal[i] / actual_weights[i]
    print(f"  Bit {resolution-1-i:2d}: {correction:.6f} (error: {(1-correction)*100:+.2f}%)")

#%% Analyze Calibrated Performance
print("\nStep 4: Analyze calibrated performance")
print("-" * 70)

fig2 = plt.figure(figsize=(12, 8))
enob_cal, sndr_cal, sfdr_cal, snr_cal, thd_cal, _, _, _ = spec_plot(
    post_cal,
    label=True,
    harmonic=5,
    osr=1,
    win_type='hamming'
)
plt.title('Calibrated ADC Output - Spectrum')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fg_cal_sine_calibrated.png'), dpi=150)
plt.close()

print(f"\nCalibrated Performance:")
print(f"  ENoB:  {enob_cal:.2f} bits")
print(f"  SNDR:  {sndr_cal:.2f} dB")
print(f"  SFDR:  {sfdr_cal:.2f} dB")

#%% Compare Before and After
print("\nStep 5: Compare before and after calibration")
print("-" * 70)

fig3, axes = plt.subplots(1, 2, figsize=(16, 6))

# Uncalibrated
plt.sca(axes[0])
spec_plot(uncalibrated_signal, label=True, harmonic=5, osr=1, win_type='hamming')
axes[0].set_title(f'Before Calibration\nENoB: {enob_uncal:.2f}, SNDR: {sndr_uncal:.2f} dB')

# Calibrated
plt.sca(axes[1])
spec_plot(post_cal, label=True, harmonic=5, osr=1, win_type='hamming')
axes[1].set_title(f'After Calibration\nENoB: {enob_cal:.2f}, SNDR: {sndr_cal:.2f} dB')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fg_cal_sine_comparison.png'), dpi=150)
plt.close()

improvement_enob = enob_cal - enob_uncal
improvement_sndr = sndr_cal - sndr_uncal

print(f"\nPerformance Improvement:")
print(f"  ENoB:  {enob_uncal:.2f} → {enob_cal:.2f} bits  (+{improvement_enob:.2f} bits)")
print(f"  SNDR:  {sndr_uncal:.2f} → {sndr_cal:.2f} dB   (+{improvement_sndr:.2f} dB)")
print(f"  SFDR:  {sfdr_uncal:.2f} → {sfdr_cal:.2f} dB   (+{sfdr_cal - sfdr_uncal:.2f} dB)")

#%% Method 2: Specify Known Frequency
print("\nStep 6: Calibration with known frequency")
print("-" * 70)

weight_cal2, offset_cal2, post_cal2, _, _, _ = fg_cal_sine(
    bits,
    freq=Fin,    # Specify known frequency
    order=5
)

enob_cal2, sndr_cal2, _, _, _, _, _, _ = spec_plot(
    post_cal2,
    label=False,
    harmonic=5,
    osr=1,
    win_type='hamming'
)
plt.close()

print(f"\nWith known frequency ({Fin:.6f}):")
print(f"  ENoB: {enob_cal2:.2f} bits")
print(f"  SNDR: {sndr_cal2:.2f} dB")
print(f"\nComparison with auto-detect:")
print(f"  ENoB difference: {abs(enob_cal2 - enob_cal):.4f} bits")
print(f"  SNDR difference: {abs(sndr_cal2 - sndr_cal):.4f} dB")

#%% Visualize Weight Errors
print("\nStep 7: Visualize weight errors and corrections")
print("-" * 70)

fig4, axes = plt.subplots(1, 2, figsize=(14, 5))

# Weight errors
bit_labels = [f'B{resolution-1-i}' for i in range(resolution)]
weight_error_pct = (actual_weights / ideal_weights - 1) * 100

axes[0].bar(bit_labels, weight_error_pct, alpha=0.7, color='red')
axes[0].set_xlabel('Bit Position (MSB to LSB)')
axes[0].set_ylabel('Weight Error (%)')
axes[0].set_title('Capacitor Mismatch (Weight Errors)')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0, color='k', linestyle='--', linewidth=0.5)

# Weight corrections
correction_pct = (weight_cal / actual_weights - 1) * 100

axes[1].bar(bit_labels, correction_pct, alpha=0.7, color='green')
axes[1].set_xlabel('Bit Position (MSB to LSB)')
axes[1].set_ylabel('Correction Factor (%)')
axes[1].set_title('Calibration Corrections Applied')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fg_cal_sine_weight_errors.png'), dpi=150)
plt.close()

#%% Summary
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(f"\nGenerated figures saved to: {output_dir}/")
print("  - fg_cal_sine_uncalibrated.png   (Before calibration)")
print("  - fg_cal_sine_calibrated.png     (After calibration)")
print("  - fg_cal_sine_comparison.png     (Side-by-side comparison)")
print("  - fg_cal_sine_weight_errors.png  (Weight errors and corrections)")

print("\nKey Takeaways:")
print("  1. fg_cal_sine estimates per-bit weights from a sinewave test")
print("  2. Set freq=0 for auto-detection, or specify exact frequency")
print("  3. Calibration corrects for capacitor mismatch and nonlinearity")
print(f"  4. In this example: {improvement_enob:.2f} bits ENoB improvement")
print("  5. Higher polynomial order (e.g., order=5) captures more nonlinearity")

print("\n" + "=" * 70)
print("Example completed successfully!")
print("=" * 70)
