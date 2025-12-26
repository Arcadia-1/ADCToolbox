"""
Batch DOUT Dashboard Generation: Digital ADC Test Cases

This example generates multiple comprehensive 6-tool dashboards for different
digital ADC scenarios. Each dashboard provides a complete diagnostic view
combining spectrum, calibration, bit activity, overflow check, ENOB sweep,
and weight radix analysis.

This is useful for:
- Creating a comprehensive test report for digital ADC characterization
- Comparing how different configurations manifest across all analysis tools
- Generating documentation and reference materials
"""

import time
from pathlib import Path

# --- 1. Timing: Imports ---
t_start = time.time()

# Set non-interactive backend before importing analysis libraries
import matplotlib
matplotlib.use('Agg')

import numpy as np
from adctoolbox.toolset import generate_dout_dashboard
from adctoolbox import find_coherent_frequency

print(f"[Timing] Library Imports: {time.time() - t_start:.4f}s")

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# --- 2. Timing: Setup ---
t_setup = time.time()

# Common parameters
N = 2**13
Fs = 1e9
Fin_target = 300e6
Fin, Fin_bin = find_coherent_frequency(fs=Fs, fin_target=Fin_target, n_fft=N)
t = np.arange(N) / Fs
A = 0.49
DC = 0.5

print(f"[Config] Fs={Fs/1e6:.0f} MHz, Fin={Fin/1e6:.2f} MHz, Bin/N=[{Fin_bin}/{N}]")
print(f"[Config] Signal: A={A:.3f}, DC={DC:.3f}")
print(f"[Timing] Setup: {time.time() - t_setup:.4f}s\n")

# --- 3. Define Test Cases ---

def generate_ideal_binary(resolution):
    """Ideal binary ADC (no noise, binary weights)."""
    signal = A * np.sin(2 * np.pi * Fin * t) + DC
    quantized = np.clip(np.floor(signal * (2**resolution)), 0, 2**resolution - 1).astype(int)
    bits = (quantized[:, None] >> np.arange(resolution - 1, -1, -1)) & 1
    return bits

def generate_with_thermal_noise(resolution, noise_level):
    """Binary ADC with thermal noise."""
    np.random.seed(42)
    noise = np.random.randn(N) * noise_level
    signal = A * np.sin(2 * np.pi * Fin * t) + DC + noise
    quantized = np.clip(np.floor(signal * (2**resolution)), 0, 2**resolution - 1).astype(int)
    bits = (quantized[:, None] >> np.arange(resolution - 1, -1, -1)) & 1
    return bits

def generate_with_lsb_random(resolution):
    """Binary ADC with random LSB (simulating digital glitch)."""
    np.random.seed(42)
    signal = A * np.sin(2 * np.pi * Fin * t) + DC
    quantized = np.clip(np.floor(signal * (2**resolution)), 0, 2**resolution - 1).astype(int)
    bits = (quantized[:, None] >> np.arange(resolution - 1, -1, -1)) & 1
    # Randomize LSB
    bits[:, -1] = np.random.randint(0, 2, N)
    return bits

def generate_sar_with_noise(resolution):
    """SAR ADC with thermal noise."""
    np.random.seed(42)
    noise = np.random.randn(N) * 200e-6
    signal = 2 * A * np.sin(2 * np.pi * Fin * t) + noise

    # Capacitor values (binary)
    caps = 2.0 ** np.arange(resolution - 1, -1, -1)
    voltage_steps = caps / np.sum(caps)

    # SAR quantization
    residue = signal.copy()
    bits = np.zeros((N, resolution))
    for j in range(resolution):
        bits[:, j] = (residue > 0).astype(int)
        if j < resolution - 1:
            delta = (2 * bits[:, j] - 1) * voltage_steps[j]
            residue -= delta

    return bits.astype(int)

# Test cases
CASES = [
    {'title': 'Ideal Binary 8-bit', 'func': lambda: generate_ideal_binary(8)},
    {'title': 'Ideal Binary 10-bit', 'func': lambda: generate_ideal_binary(10)},
    {'title': 'Ideal Binary 12-bit', 'func': lambda: generate_ideal_binary(12)},
    {'title': 'Binary 12-bit + Thermal Noise (50uV)', 'func': lambda: generate_with_thermal_noise(12, 50e-6)},
    {'title': 'Binary 12-bit + Thermal Noise (200uV)', 'func': lambda: generate_with_thermal_noise(12, 200e-6)},
    {'title': 'Binary 12-bit + LSB Random', 'func': lambda: generate_with_lsb_random(12)},
    {'title': 'SAR 12-bit + Thermal Noise', 'func': lambda: generate_sar_with_noise(12)},
]

# --- 4. Generate Dashboards ---
t_batch = time.time()

print("\n" + "=" * 100)
print(f"{'#':<4} | {'Test Case':<40} | {'Status':<40} | {'Time (s)':<10}")
print("-" * 100)

for idx, case in enumerate(CASES):
    t_case = time.time()

    # Generate digital bits
    bits = case['func']()

    # Create dashboard filename
    safe_title = case['title'].replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'with')
    fig_path = (output_dir / f'exp_t04_dashboard_{idx+1:02d}_{safe_title}.png').resolve()

    # Generate dashboard
    fig, axes = generate_dout_dashboard(
        bits=bits,
        freq=Fin/Fs,  # Normalized frequency
        weights=None,  # Use binary weights by default
        output_path=fig_path
    )

    elapsed = time.time() - t_case
    print(f"{idx+1:<4} | {case['title']:<40} | {fig_path.name:<40} | {elapsed:<10.3f}")

print("=" * 100)
print(f"\n[Timing] Batch Dashboard Generation: {time.time() - t_batch:.4f}s")

print(f"\n--- Total Runtime: {time.time() - t_start:.4f}s ---\n")

print("\n" + "=" * 100)
print("Dashboard Generation Complete!")
print("-" * 100)
print(f"Generated {len(CASES)} comprehensive 6-tool dashboards")
print(f"Each dashboard includes:")
print("  1. Spectrum: Nominal Weights")
print("  2. Spectrum: Calibrated Weights")
print("  3. Bit Activity")
print("  4. Overflow Check")
print("  5. ENOB Bit Sweep")
print("  6. Weight Radix")
print(f"\nOutput directory: {output_dir}")
print("=" * 100)
