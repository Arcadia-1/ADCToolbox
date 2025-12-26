"""
Overflow Check: Residue Distribution Analysis for Sub-Radix SAR ADC

This example demonstrates the analyze_overflow function for visualizing residue
distributions in SAR ADC digital outputs. The residue plot shows how close each
bit segment gets to its limits [0, 1].

Four Test Cases:
1. Binary ADC, Normal Range - Residue well within bounds
2. Binary ADC, Large Signal - Residue touches boundaries (normal clipping)
3. Sub-Radix with Redundancy - Redundancy provides safety margin
4. Sub-Radix, Insufficient Redundancy - Reduced safety margin

Key Insights:
- Blue points: Samples within normal range
- Red/Yellow points: Samples touching boundaries (not necessarily overflow!)
- Red envelope: Shows min/max residue range
- Percentages: % of samples touching boundaries (informational only)
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import freq_to_bin, calibrate_weight_sine, analyze_overflow

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Timing start
t_start = time.time()

# Signal generation
n_samples = 2**13
fs = 1e9
bin = freq_to_bin(300e6, fs, n_samples)
fin = (bin / n_samples) * fs
t = np.arange(n_samples) / fs

# Test cases: [caps, amplitude, title]
# caps defines the capacitor DAC weights (determines redundancy)
test_cases = [
    # Case 1: Binary, normal amplitude (no overflow expected)
    {
        'caps': np.array([1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 1], dtype=float),
        'amplitude': 0.49,
        'title': 'Binary ADC, Normal Range',
        'description': 'Ideal case: no overflow'
    },

    # Case 2: Binary, large amplitude (overflow at MSB expected)
    {
        'caps': np.array([1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 1], dtype=float),
        'amplitude': 0.55,
        'title': 'Binary ADC, Large Signal',
        'description': 'Signal clips, MSB overflow'
    },

    # Case 3: Sub-radix with good redundancy (no overflow expected)
    {
        'caps': np.array([1024, 512, 256, 256, 128, 64, 32, 16, 8, 4, 2, 1, 1], dtype=float),
        'amplitude': 0.49,
        'title': 'Sub-Radix with Redundancy',
        'description': 'Bit 3-4 redundant, prevents overflow'
    },

    # Case 4: Sub-radix with insufficient redundancy (overflow expected)
    {
        'caps': np.array([1024, 512, 256, 200, 128, 64, 32, 16, 8, 4, 2, 1, 1], dtype=float),
        'amplitude': 0.49,
        'title': 'Sub-Radix, Insufficient Redundancy',
        'description': 'Bit 4 too small, causes overflow'
    },
]

fig, axes = plt.subplots(1, 4, figsize=(24, 5))

for idx, case in enumerate(test_cases):
    caps = case['caps']
    amplitude = case['amplitude']
    title = case['title']
    n_bits = len(caps)

    # Generate signal
    signal = 2 * amplitude * np.sin(2 * np.pi * fin * t)

    # Hardware simulation
    voltage_steps = caps / np.sum(caps)

    # SAR quantization
    residue = signal.copy()
    digital_output = np.zeros((n_samples, n_bits))
    for j in range(n_bits):
        digital_output[:, j] = (residue > 0).astype(int)
        if j < n_bits - 1:
            delta = (2 * digital_output[:, j] - 1) * voltage_steps[j]
            residue -= delta

    # Calibrate to get weights
    cal_result = calibrate_weight_sine(digital_output, freq=bin / n_samples)
    weights_calibrated = cal_result['weight']

    # Run overflow check
    t_ovf = time.time()
    range_min, range_max, ovf_pct_zero, ovf_pct_one = analyze_overflow(
        digital_output,
        weights_calibrated,
        create_plot=True,
        ax=axes[idx]
    )
    elapsed_ovf = time.time() - t_ovf

    axes[idx].set_title(f'{title}\n{case["description"]}',
                        fontsize=12, fontweight='bold')

    # Print summary
    range_span = range_max[0] - range_min[0]  # MSB range
    print(f"[{title:40s}] Runtime={elapsed_ovf*1e3:6.2f}ms, "
          f"MSB_range=[{range_min[0]:.3f}, {range_max[0]:.3f}], "
          f"Span={range_span:.3f}")

plt.tight_layout()
fig_path = output_dir / 'exp_d14_overflow_check.png'
plt.savefig(fig_path, dpi=150)
print(f"\n[Save fig] -> [{fig_path}]")

elapsed_total = time.time() - t_start
print(f"\n--- Total Runtime: {elapsed_total:.4f}s ---\n")

plt.close('all')
