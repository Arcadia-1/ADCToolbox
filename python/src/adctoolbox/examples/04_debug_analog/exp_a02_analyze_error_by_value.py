"""
Demonstrates `analyze_error_by_value` for distinguishing thermal noise from static nonlinearity.
This method provides a quick, coarse visualization of the INL shape (error vs. code)
to identify static nonlinearity errors without running a full histogram test.

3 Figures:
- Figure 1: Thermal Noise (50 bins)
- Figure 2: 3rd Order Nonlinearity (50 bins)
- Figure 3: 3rd Order Nonlinearity (200 bins) - Higher resolution
"""

import time

# --- 1. Timing: Imports ---
t_start = time.time()
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import analyze_error_by_value

print(f"[Timing] Library Imports: {time.time() - t_start:.4f}s")

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# --- 2. Timing: Data Generation ---
t_gen = time.time()

# Setup parameters
N = 2**13
Fs = 800e6
Fin = 10.1234567e6
normalized_freq = Fin / Fs
t = np.arange(N) / Fs
A = 0.49
DC = 0.5
base_noise = 50e-6
print(f"[Config] Fs={Fs/1e6:.0f} MHz, Fin={Fin/1e6:.2f} MHz, N={N}")

# Case 1: Ideal ADC with Thermal Noise
sig_noise = A * np.sin(2 * np.pi * Fin * t) + DC + np.random.randn(N) * base_noise

# Case 2: ADC with 3rd Order Nonlinearity
k3 = 0.01
sig_nonlin = A * np.sin(2 * np.pi * Fin * t) + DC + k3 * (A * np.sin(2 * np.pi * Fin * t))**3 + np.random.randn(N) * base_noise
print(f"[Timing] Data Generation: {time.time() - t_gen:.4f}s")

# --- 3. Timing: Analysis & Plotting (InMemory) ---
t_plot = time.time()

# Analyze and plot results with 3 figures
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
fig.suptitle('Value Error Analysis - Thermal Noise vs 3rd Order Nonlinearity', fontsize=14, fontweight='bold')

analyze_error_by_value(sig_noise, n_bins=16, ax=axes[0], title='Thermal Noise Only')
analyze_error_by_value(sig_nonlin, n_bins=16, ax=axes[1], title='3rd Order Nonlinearity (16 bins)')
analyze_error_by_value(sig_nonlin, n_bins=64, ax=axes[2], title='3rd Order Nonlinearity (64 bins)')

plt.tight_layout()

print(f"[Timing] Analysis & Plotting Setup: {time.time() - t_plot:.4f}s")

# --- 4. Timing: File Saving (Rendering) ---
t_save = time.time()

fig_path_bins = (output_dir / 'exp_a02_analyze_error_by_value_bins.png').resolve()
plt.savefig(fig_path_bins, dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"[Save fig] -> [{fig_path_bins}]")
print(f"[Timing] Image Rendering & Saving: {time.time() - t_save:.4f}s")

print(f"--- Total Runtime: {time.time() - t_start:.4f}s ---\n")