"""Phase-based error analysis: Raw data approach

This example demonstrates the analyze_error_by_phase wrapper function for detecting
phase-dependent noise (AM/PM decomposition) using raw data method.

Compares two cases:
1. Ideal ADC with thermal noise (50 uVrms)
2. ADC with phase jitter (0.05 rad RMS)
"""

import time

# --- 1. Timing: Imports ---
t_start = time.time()
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import analyze_error_by_phase

print(f"[Timing] Library Imports: {time.time() - t_start:.4f}s")

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# --- 2. Timing: Data Generation ---
t_gen = time.time()

# Setup parameters
N = 2**16
Fs = 800e6
Fin = 10.1234567e6  # no need to be coherent for phase error analysis
normalized_freq = Fin / Fs
t = np.arange(N) / Fs
A = 0.49
base_noise = 50e-6
print(f"[Config] Fs={Fs/1e6:.0f} MHz, Fin={Fin/1e6:.2f} MHz, N={N}")

# Case 1: Ideal ADC with Thermal Noise
sig_noise = A * np.sin(2 * np.pi * Fin * t) + np.random.randn(N) * base_noise

# Case 2: ADC with Phase Jitter
phase_jitter = 0.05 * np.random.randn(N)  # 0.05 rad RMS
jitter_phase = 2 * np.pi * Fin * t + phase_jitter
sig_jitter = A * np.sin(jitter_phase) + np.random.randn(N) * base_noise

print(f"[Timing] Data Generation: {time.time() - t_gen:.4f}s")

# --- 3. Timing: Analysis & Plotting (Raw Mode) ---
t_plot = time.time()

# Analyze and plot results - Raw mode (with raw scatter plot)
fig = plt.figure(figsize=(16, 8))
fig.suptitle('Phase Error Analysis (Raw Mode) - Thermal Noise vs Phase Jitter', fontsize=16, fontweight='bold')

ax1 = plt.subplot(1, 2, 1)
analyze_error_by_phase(sig_noise, normalized_freq, mode="raw", show_plot=True, plot_mode="raw", ax=ax1)

ax2 = plt.subplot(1, 2, 2)
analyze_error_by_phase(sig_jitter, normalized_freq, mode="raw", show_plot=True, plot_mode="raw", ax=ax2)

plt.tight_layout()

print(f"[Timing] Analysis & Plotting Setup: {time.time() - t_plot:.4f}s")

# --- 4. Timing: File Saving (Rendering) ---
t_save = time.time()

fig_path = (output_dir / 'exp_a11_analyze_error_by_phase_raw.png').resolve()
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"[Save fig] -> [{fig_path}]")
print(f"[Timing] Image Rendering & Saving: {time.time() - t_save:.4f}s")

print(f"--- Total Runtime: {time.time() - t_start:.4f}s ---\n")
