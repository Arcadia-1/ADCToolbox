"""
Comprehensive Digital ADC Analysis Dashboard: Single calibration case with 6 diagnostic plots.

This example demonstrates using the generate_dout_dashboard function
to create a 2x3 panel showing 6 different analysis perspectives of digital bit data.
Uses a quantized sinewave to show how each analysis tool reveals different aspects
of ADC digital performance.

6 Analysis Tools:
1. analyze_spectrum - Spectrum with nominal weights (binary by default)
2. analyze_spectrum - Spectrum after calibration
3. analyze_bit_activity - Bit usage percentage (ideal = 50%)
4. analyze_overflow - Residue distribution for overflow detection
5. analyze_enob_sweep - ENOB vs number of bits used
6. analyze_weight_radix - Weight radix visualization
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

# --- 2. Timing: Data Generation ---
t_gen = time.time()

# Setup parameters
N = 2**13
Fs = 1e9
Fin_target = 300e6
Fin, Fin_bin = find_coherent_frequency(fs=Fs, fin_target=Fin_target, n_fft=N)
t = np.arange(N) / Fs
A = 0.49
DC = 0.5
resolution = 12

print(f"[Config] Fs={Fs/1e6:.0f} MHz, Fin={Fin/1e6:.2f} MHz, Bin/N=[{Fin_bin}/{N}], Resolution={resolution} bits")
print(f"[Config] Signal: A={A:.3f}, DC={DC:.3f}")

# Generate quantized sinewave
signal = A * np.sin(2 * np.pi * Fin * t) + DC
quantized_signal = np.clip(np.floor(signal * (2**resolution)), 0, 2**resolution - 1).astype(int)

# Extract digital bits (MSB to LSB)
bits = (quantized_signal[:, None] >> np.arange(resolution - 1, -1, -1)) & 1

print(f"[Timing] Data Generation: {time.time() - t_gen:.4f}s")

# --- 3. Timing: Analysis & Plotting & Saving (Dashboard Generation) ---
t_plot = time.time()

fig_path = (output_dir / 'exp_t03_dout_dashboard.png').resolve()

# Generate dashboard with auto-frequency detection (freq=None)
fig, axes = generate_dout_dashboard(
    bits=bits,
    freq=Fin/Fs,  # Normalized frequency
    weights=None,  # Use binary weights by default
    output_path=fig_path
)

print(f"[Timing] Dashboard Generation & Saving: {time.time() - t_plot:.4f}s")

print(f"\n--- Total Runtime: {time.time() - t_start:.4f}s ---\n")
