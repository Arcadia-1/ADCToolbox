import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
from adctoolbox import freq_to_bin, analyze_enob_sweep

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Signal generation
n_samples = 2**13
fs = 1e9
bin = freq_to_bin(300e6, fs, n_samples)
fin = (bin / n_samples) * fs
amplitude = 0.499

# Hardware setup - Binary ADC
caps_nominal = np.array([1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 1], dtype=float)
n_bits = len(caps_nominal)
nominal_resolution = np.log2(np.sum(caps_nominal) / caps_nominal[-1] * 2)
voltage_steps = caps_nominal / np.sum(caps_nominal)

# Test cases: [lsb_random, title]
test_cases = [
    (False, 'Binary ADC with Thermal Noise'),
    (True, 'Binary ADC with Thermal Noise + LSB random'),
]

# Set font sizes for all plot elements
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (lsb_random, title) in enumerate(test_cases):
    # Signal with thermal noise
    np.random.seed(42)
    noise = np.random.randn(n_samples) * 200e-6
    signal = 2 * amplitude * np.sin(2 * np.pi * fin * np.arange(n_samples) / fs) + noise

    # SAR quantization
    residue = signal.copy()
    digital_output = np.zeros((n_samples, n_bits))
    for j in range(n_bits):
        digital_output[:, j] = (residue > 0).astype(int)
        if j < n_bits - 1:
            delta = (2 * digital_output[:, j] - 1) * voltage_steps[j]
            residue -= delta

    # Add LSB glitch (completely random)
    if lsb_random:
        digital_output[:, -1] = np.random.randint(0, 2, n_samples)

    # Run ENOB bit sweep
    start_time = time.time()
    enob_sweep, n_bits_vec = analyze_enob_sweep(
        digital_output, freq=fin/fs, harmonic_order=1, osr=1, win_type='hamming', create_plot=True, ax=axes[idx]
    )
    elapsed_time = time.time() - start_time
    axes[idx].set_title(title, fontweight='bold')

    final_enob = enob_sweep[-1]
    max_enob = np.max(enob_sweep)
    optimal_bits = n_bits_vec[np.argmax(enob_sweep)]

    print(f"[{title:45s}] Runtime={elapsed_time*1e3:7.2f}ms, "
          f"Nominal={nominal_resolution:5.2f} bit, "
          f"Max_ENOB={max_enob:5.2f} bit @ {optimal_bits} bits, "
          f"Final_ENOB={final_enob:5.2f} bit")

plt.tight_layout()
fig_path = output_dir / 'exp_d12_sweep_bit_enob.png'
plt.savefig(fig_path, dpi=150)
print(f"\n[Save fig] -> [{fig_path}]")
plt.close('all')
