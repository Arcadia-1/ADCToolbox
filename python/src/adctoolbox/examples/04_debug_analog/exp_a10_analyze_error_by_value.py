"""
Demonstrates `analyze_error_by_value` for distinguishing thermal noise from static nonlinearity.
This method provides a quick, coarse visualization of the INL shape (error vs. code)
to identify static nonlinearity errors without running a full histogram test.
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
N = 2**16
Fs = 800e6
Fin = 10.1234567e6
normalized_freq = Fin / Fs
t = np.arange(N) / Fs
A = 0.49
base_noise = 50e-6
print(f"[Config] Fs={Fs/1e6:.0f} MHz, Fin={Fin/1e6:.2f} MHz, N={N}")

# Case 1: Ideal ADC with Thermal Noise
sig_noise = A * np.sin(2 * np.pi * Fin * t) + np.random.randn(N) * base_noise

# Case 2: ADC with 3rd Order Nonlinearity
k3 = 0.01
sig_nonlin = A * np.sin(2 * np.pi * Fin * t) + k3 * (A * np.sin(2 * np.pi * Fin * t))**3 + np.random.randn(N) * base_noise

print(f"[Timing] Data Generation: {time.time() - t_gen:.4f}s")

# --- 3. Timing: Analysis & Plotting (InMemory) ---
t_plot = time.time()

# Analyze and plot results
fig = plt.figure(figsize=(16, 8))
fig.suptitle('Value Error Analysis (Coarse INL Check) - Thermal Noise vs 3rd Order Nonlinearity', fontsize=16, fontweight='bold')

ax1 = plt.subplot(1, 2, 1)
analyze_error_by_value(sig_noise, n_bins=50, ax=ax1)

ax2 = plt.subplot(1, 2, 2)
analyze_error_by_value(sig_nonlin, n_bins=50, ax=ax2)

plt.tight_layout()

print(f"[Timing] Analysis & Plotting Setup: {time.time() - t_plot:.4f}s")

# --- 4. Timing: File Saving (Rendering) ---
t_save = time.time()

fig_path_bins = (output_dir / 'exp_a10_analyze_error_by_value_bins.png').resolve()
plt.savefig(fig_path_bins, dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"[Save fig] -> [{fig_path_bins}]")
print(f"[Timing] Image Rendering & Saving: {time.time() - t_save:.4f}s")

print(f"--- Total Runtime: {time.time() - t_start:.4f}s ---\n")