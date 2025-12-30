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
amplitude = 0.49
signal = 2 * amplitude * np.sin(2 * np.pi * fin * np.arange(n_samples) / fs)

# Test cases: [caps_nominal, mismatch_factor, title]
test_cases = [
    (np.array([1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 1], dtype=float), 0.98, 'No Redundancy (MSB -2%)'),
    (np.array([1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 1], dtype=float), 1.02, 'No Redundancy (MSB +2%)'),
    (np.array([1024, 512, 256, 256, 128, 64, 32, 16, 8, 4, 2, 1, 1], dtype=float), 0.98, 'With Redundancy (MSB -2%)'),
    (np.array([1024, 512, 256, 256, 128, 64, 32, 16, 8, 4, 2, 1, 1], dtype=float), 1.02, 'With Redundancy (MSB +2%)'),
]

fig, axes = plt.subplots(2, 4, figsize=(20, 9))
results = []

for idx, (caps_nominal, mismatch_factor, title) in enumerate(test_cases):
    n_bits = len(caps_nominal)
    weights_nominal = np.append(caps_nominal[:-1], caps_nominal[-1] * 0.5)
    nominal_resolution = np.log2(np.sum(caps_nominal) / caps_nominal[-1] * 2)

    # Real hardware with mismatch (first bit has error)
    caps_real = caps_nominal.copy()
    caps_real[0] *= mismatch_factor
    voltage_steps = caps_real / np.sum(caps_real)

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
    cal_results = calibrate_weight_sine(digital_output, freq=bin / n_samples)
    analog_after = cal_results["calibrated_signal"][0]
    weights_calibrated = cal_results["weight"]

    # Spectrum comparison
    plt.sca(axes[0, idx])
    result_before = analyze_spectrum(analog_before, max_harmonic=5, show_label=True, ax=axes[0, idx])
    axes[0, idx].set_title(f'{title}\nBefore Calibration', fontsize=11, fontweight='bold')

    plt.sca(axes[1, idx])
    result_after = analyze_spectrum(analog_after, max_harmonic=5, show_label=True, ax=axes[1, idx])
    axes[1, idx].set_title(f'After Calibration', fontsize=11, fontweight='bold')

    # Normalize weights for error analysis
    weights_real = np.append(caps_real[:-1], caps_real[-1] * 0.5)
    weights_nominal_norm = weights_nominal / (np.sum(weights_nominal) + weights_nominal[-1])
    weights_real_norm = weights_real / (np.sum(weights_real) + weights_real[-1])
    weights_calibrated_norm = weights_calibrated / (np.sum(weights_calibrated) + weights_calibrated[-1])

    error_before = np.abs(weights_nominal_norm - weights_real_norm)
    error_after = np.abs(weights_calibrated_norm - weights_real_norm)

    results.append({
        'title': title,
        'n_bits': n_bits,
        'nominal_resolution': nominal_resolution,
        'enob_before': result_before['enob'],
        'enob_after': result_after['enob'],
        'error_before': error_before,
        'error_after': error_after,
        'max_error_before': np.max(error_before),
        'max_error_after': np.max(error_after),
    })

    print(f"[{title:35s}] [Nominal = {nominal_resolution:5.2f} bit] [ENoB = {result_before['enob']:5.2f} bit] -> [ENoB = {result_after['enob']:5.2f} bit]")

plt.tight_layout()
fig_path = output_dir / 'exp_d03_redundancy_comparison.png'
plt.savefig(fig_path, dpi=150)
print(f"\n[Save fig] -> [{fig_path}]")

# Weight error comparison plot
fig, axes = plt.subplots(2, 2, figsize=(12, 6))

for idx, result in enumerate(results):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    bit_indices = np.arange(result['n_bits'])
    width = 0.35

    ax.bar(bit_indices - width/2, result['error_before'], width,
           label='Before Calibration', alpha=0.7, color='red')
    ax.bar(bit_indices + width/2, result['error_after'], width,
           label='After Calibration', alpha=0.7, color='green')

    ax.set_title(f"{result['title']}\n"
                 f"Max Error: {result['max_error_before']:.6f} → {result['max_error_after']:.6f}",
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Bit Index (0=MSB)', fontsize=10)
    ax.set_ylabel('Absolute Weight Error', fontsize=10)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y', which='both')
    ax.legend()

plt.tight_layout()
fig_path_error = output_dir / 'exp_d03_weight_error_comparison.png'
plt.savefig(fig_path_error, dpi=150)
print(f"[Save fig] -> [{fig_path_error}]")
plt.close('all')
