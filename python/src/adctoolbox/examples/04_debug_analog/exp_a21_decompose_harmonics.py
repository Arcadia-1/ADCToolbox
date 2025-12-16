"""
Demonstrates `analyze_decomposition_time` for distinguishing thermal noise from static nonlinearity.
This method provides quick visualization of harmonic decomposition to identify nonlinearity
without running full FFT-based spectral analysis.
"""

import time

# --- 1. Timing: Imports ---
t_start = time.time()
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import analyze_decomposition_time

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
A = 0.499
DC = 0.5
base_noise = 50e-6

sig_ideal = A * np.sin(2 * np.pi * Fin * t) + DC
print(f"[Config] Fs={Fs/1e6:.0f} MHz, Fin={Fin/1e6:.2f} MHz, N={N}")

# Case 1: Ideal ADC with Thermal Noise
sig_noise = sig_ideal + np.random.randn(N) * base_noise

# Case 2: ADC with Nonlinearity (k2 and k3)
k2 = 0.001
k3 = 0.005
sig_nonlin = sig_ideal + k2 * sig_ideal**2 + k3 * sig_ideal**3 + np.random.randn(N) * base_noise

print(f"[Timing] Data Generation: {time.time() - t_gen:.4f}s")

# --- 3. Timing: Analysis & Plotting (In-Memory) ---
t_plot = time.time()

# Analyze and plot results
fig = plt.figure(figsize=(16, 8))
fig.suptitle('Harmonic Decomposition - Thermal Noise vs Static Nonlinearity', fontsize=16, fontweight='bold')

ax1 = plt.subplot(1, 2, 1)
analyze_decomposition_time(sig_noise, harmonic=3, fs=Fs, show_plot=True, ax=ax1)
ax1.set_title('Case 1: Thermal Noise (50μV RMS)', fontweight='bold')

ax2 = plt.subplot(1, 2, 2)
analyze_decomposition_time(sig_nonlin, harmonic=3, fs=Fs, show_plot=True, ax=ax2)
ax2.set_title(f'Case 2: Static Nonlinearity (k2={k2:.3f}, k3={k3:.3f})', fontweight='bold')

plt.tight_layout()

print(f"[Timing] Analysis & Plotting Setup: {time.time() - t_plot:.4f}s")

# --- 4. Timing: File Saving (Rendering) ---
t_save = time.time()

fig_path = (output_dir / 'exp_a21_decompose_harmonics.png').resolve()
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"[Save fig] -> [{fig_path}]")
print(f"[Timing] Image Rendering & Saving: {time.time() - t_save:.4f}s")

print(f"--- Total Runtime: {time.time() - t_start:.4f}s ---\n")
