"""
Lightweight Weight Calibration: Before/After Spectrum Comparison

Demonstrates calibrate_weight_sine_lite for fast ADC weight calibration.
Shows spectrum before and after calibration overlapped on the same plot.
Uncalibrated spectrum is shown without labels, calibrated spectrum with labels.

This allows easy visual comparison of calibration improvement.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import freq_to_bin, analyze_spectrum
from adctoolbox.calibration import calibrate_weight_sine_lite

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Timing start
t_start = time.time()

# Signal generation
n_samples = 2**13
fs = 1e9
bin = freq_to_bin(80e6, fs, n_samples)
fin = (bin / n_samples) * fs
freq_norm = fin / fs  # Normalized frequency
amplitude = 0.499

print(f"[Config] N={n_samples}, Fs={fs/1e6:.0f} MHz, Fin={fin/1e6:.2f} MHz, Bin={bin}")

# Hardware setup with 1% weight mismatch
bit_width = 12
caps_nominal = 2.0 ** np.arange(bit_width - 1, -1, -1)

# Real hardware with 1% error on MSB (simulates capacitor mismatch)
caps_real = caps_nominal.copy()
caps_real[0] *= 0.99  # MSB 1% smaller
voltage_steps = caps_real / np.sum(caps_real)

# Generate analog signal
t = np.arange(n_samples)
signal = 2 * amplitude * np.sin(2 * np.pi * freq_norm * t + np.pi/4)

# SAR ADC quantization with real (mismatched) weights
residue = signal.copy()
bits = np.zeros((n_samples, bit_width), dtype=int)
for j in range(bit_width):
    bits[:, j] = (residue > 0).astype(int)
    if j < bit_width - 1:
        delta = (2 * bits[:, j] - 1) * voltage_steps[j]
        residue -= delta

# Reconstruct with nominal weights (creates error due to mismatch)
true_weights = caps_nominal
quantized_signal = bits @ true_weights

print(f"[Signal] Bit_width={bit_width}, MSB_mismatch=-1%, Quantized range=[{quantized_signal.min():.0f}, {quantized_signal.max():.0f}]")

# Run lightweight calibration
t_cal = time.time()
recovered_weights = calibrate_weight_sine_lite(bits, freq=freq_norm)
elapsed_cal = time.time() - t_cal

# Scale recovered weights to match true weights range
recovered_weights_scaled = recovered_weights * np.max(true_weights)

# Compute calibrated signal and ideal reference
calibrated_signal = bits @ recovered_weights_scaled
adc_amplitude = (2**bit_width - 1) / 2.0
ideal_signal = adc_amplitude * np.sin(2 * np.pi * freq_norm * t + np.pi/8) + adc_amplitude
error_signal = calibrated_signal - ideal_signal

# Create overlapping spectrum plot
fig, ax = plt.subplots(figsize=(8, 6))

# Plot uncalibrated spectrum (no labels)
result_before = analyze_spectrum(
    quantized_signal,
    fs=fs,
    max_harmonic=5,
    show_label=False,
    ax=ax,
    create_plot=True
)

# Change uncalibrated spectrum to gray
for line in ax.get_lines():
    line.set_color('gray')
    line.set_alpha(0.5)
    line.set_linewidth(1)

# Reset color cycle to default blue (C0)
ax.set_prop_cycle(None)

# Plot calibrated spectrum (with labels)
result_after = analyze_spectrum(
    calibrated_signal,
    fs=fs,
    max_harmonic=5,
    show_label=True,
    ax=ax,
    create_plot=True
)

# Extract SNDR/ENOB from plotting results (reuse instead of recalculating)
sndr_before = result_before['sndr_dbc']
enob_before = result_before['enob']
sndr_after = result_after['sndr_dbc']
enob_after = result_after['enob']

# Calculate SNDR manually for verification
sndr_calc = 10 * np.log10(np.mean(ideal_signal**2) / np.mean(error_signal**2))
enob_calc = (sndr_calc - 1.76) / 6.02

# Print spectrum metrics
print(f"\n[Spectrum Before] ENOB={result_before['enob']:5.2f} b, SNDR={result_before['sndr_dbc']:6.2f} dB, SFDR={result_before['sfdr_dbc']:6.2f} dB, SNR={result_before['snr_dbc']:6.2f} dB, NSD={result_before['nsd_dbfs_hz']:7.2f} dBFS/Hz")
print(f"[Spectrum Before] Noise Floor={result_before['noise_floor_dbfs']:7.2f} dBFS, Signal Power={result_before['sig_pwr_dbfs']:6.2f} dBFS")

print(f"\n[Spectrum After]  ENOB={result_after['enob']:5.2f} b, SNDR={result_after['sndr_dbc']:6.2f} dB, SFDR={result_after['sfdr_dbc']:6.2f} dB, SNR={result_after['snr_dbc']:6.2f} dB, NSD={result_after['nsd_dbfs_hz']:7.2f} dBFS/Hz")
print(f"[Spectrum After]  Noise Floor={result_after['noise_floor_dbfs']:7.2f} dBFS, Signal Power={result_after['sig_pwr_dbfs']:6.2f} dBFS")

# Print weight comparison
true_weights_str = ', '.join([f'{w:5.1f}' for w in true_weights])
recovered_weights_str = ', '.join([f'{w:5.1f}' for w in recovered_weights_scaled])
print(f"\n[Weight Recovery]")
print(f"  True weights     : [{true_weights_str}]")
print(f"  Recovered weights: [{recovered_weights_str}]")

# Print SNDR/ENOB comparison
print(f"\n[Performance]")
print(f"  Calibration Runtime: {elapsed_cal*1e3:.2f} ms")
print(f"  SNDR: {sndr_before:.2f} dB -> {sndr_after:.2f} dB (calc: {sndr_calc:.2f} dB)")
print(f"  ENOB: {enob_before:.2f} bit -> {enob_after:.2f} bit (calc: {enob_calc:.2f} bit)")
print(f"  Improvement: +{sndr_after - sndr_before:.2f} dB, +{enob_after - enob_before:.2f} bit")

# Set title with performance metrics
ax.set_title(
    f'Lightweight Calibration: Before vs After\n'
    f'ENOB: {enob_before:.2f} -> {enob_after:.2f} bit (+{enob_after - enob_before:.2f} bit)',
    fontsize=12, fontweight='bold'
)

ax.set_ylim(bottom=-140)
plt.tight_layout()

fig_path = output_dir / 'exp_d01_cal_weight_sine_lite.png'
plt.savefig(fig_path, dpi=150)
print(f"\n[Save fig] -> [{fig_path}]")

elapsed_total = time.time() - t_start
print(f"\n--- Total Runtime: {elapsed_total:.4f}s ---\n")

plt.close('all')
