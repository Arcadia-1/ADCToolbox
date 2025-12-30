import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import freq_to_bin, calibrate_weight_sine, analyze_spectrum

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Signal generation
n_samples = 2**13
fs = 1e9
bin = freq_to_bin(300e6, fs, n_samples)
fin = (bin / n_samples) * fs
amplitude = 0.499
signal = 2 * amplitude * np.sin(2 * np.pi * fin * np.arange(n_samples) / fs)

# Hardware setup
caps_nominal = np.array([1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 1], dtype=float)
n_bits = len(caps_nominal)

# Real hardware with mismatch
caps_real = caps_nominal.copy()
caps_real[0] *= 0.99  # MSB a little bit smaller, you can change these values to test
voltage_steps = caps_real / np.sum(caps_real)

# Nominal weights for reconstruction (normalized to voltage range)
# For a SAR ADC, weights should sum to ~1.0 (full scale)
weights_nominal = caps_nominal / np.sum(caps_nominal)
weights_nominal[-1] = weights_nominal[-1] / 2  # Last bit is comparator, half weight

# SAR quantization
residue = signal.copy()
digital_output = np.zeros((n_samples, n_bits))
for j in range(n_bits):
    digital_output[:, j] = (residue > 0).astype(int)
    if j < n_bits - 1:
        delta = (2 * digital_output[:, j] - 1) * voltage_steps[j]
        residue -= delta

# Reconstruction before calibration
analog_before = np.dot(digital_output, weights_nominal)

# Calibration
results = calibrate_weight_sine(digital_output, freq=bin / n_samples)
analog_after = results['calibrated_signal'][0]
weights_calibrated = results['weight']

# Spectrum comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

result_before = analyze_spectrum(analog_before, max_harmonic=5, show_label=True, ax=axes[0])
axes[0].set_title('Before Calibration', fontsize=12, fontweight='bold')
axes[0].set_ylim(bottom= -140)

result_after = analyze_spectrum(analog_after, max_harmonic=5, show_label=True, ax=axes[1])
axes[1].set_title('After Calibration', fontsize=12, fontweight='bold')
axes[1].set_ylim(bottom= -140)

# Normalize weights for comparison
weights_real = caps_real / np.sum(caps_real)
weights_real[-1] = weights_real[-1] / 2  # Last bit is comparator

# calibrate_weight_sine returns weights that sum to ~2.0 (differential signal)
# Normalize to match the single-ended weights (sum ~1.0)
weights_nominal_norm = weights_nominal
weights_real_norm = weights_real
weights_calibrated_norm = weights_calibrated / 2.0  # Convert differential to single-ended

# Print results
print(f"[Nominal Resolution] {n_bits} bits")
print(f"[Weight Calibration] [ENoB = {result_before['enob']:5.2f} bit] -> [ENoB = {result_after['enob']:5.2f} bit]")

nominal_str = ', '.join([f'{w:.5f}' for w in weights_nominal_norm])
weights_str = ', '.join([f'{w:.5f}' for w in weights_real_norm])
calibrated_str = ', '.join([f'{w:.5f}' for w in weights_calibrated_norm])
print(f"  [Nominal]: [{nominal_str}]")
print(f"  [Real   ]: [{weights_str}] <-- Truth")
print(f"  [Cal    ]: [{calibrated_str}] <-- Result")

plt.tight_layout()
fig_path = output_dir / 'exp_d02_cal_weight_sine.png'
plt.savefig(fig_path, dpi=150)
print(f"\n[Save fig] -> [{fig_path}]")

# Calibration error analysis (absolute fractional error)
error_before = np.abs(weights_nominal_norm - weights_real_norm)
error_after = np.abs(weights_calibrated_norm - weights_real_norm)

fig, ax = plt.subplots(figsize=(8, 4))
bit_indices = np.arange(n_bits)
width = 0.35

ax.bar(bit_indices - width/2, error_before, width, label='Before Calibration', alpha=0.7, color='red')
ax.bar(bit_indices + width/2, error_after, width, label='After Calibration', alpha=0.7, color='green')

max_error_before = np.max(error_before)
max_error_after = np.max(error_after)
ax.set_title(f'Weight Error: Before vs After Calibration\n'
             f'Max Error: {max_error_before:.6f} → {max_error_after:.6f}',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Bit Index (0=MSB)', fontsize=11)
ax.set_ylabel('Absolute Weight Error', fontsize=11)
ax.set_yscale('log')
ax.grid(True, alpha=0.3, axis='y', which='both')
ax.legend()
plt.tight_layout()
fig_path_error = output_dir / 'exp_d02_weight_error_comparison.png'
plt.savefig(fig_path_error, dpi=150)
print(f"[Save fig] -> [{fig_path_error}]")
plt.close('all')
