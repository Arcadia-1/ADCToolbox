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
A = 0.25
DC = 0.5
base_noise = 50e-6

sig_ac = A * np.sin(2 * np.pi * Fin * t)  # AC component only
sig_ideal = sig_ac + DC
print(f"[Config] Fs={Fs/1e6:.0f} MHz, Fin={Fin/1e6:.2f} MHz, N={N}")

# Case 1: Ideal ADC with Thermal Noise
sig_noise = sig_ideal + np.random.randn(N) * base_noise

# Case 2: ADC with Nonlinearity (k2 and k3 applied to AC component only)
k2 = 0.001
k3 = 0.005
sig_nonlin = DC + sig_ac + k2 * sig_ac**2 + k3 * sig_ac**3 + np.random.randn(N) * base_noise
 
# Case 3: ADC with Glitches
glitch_prob = 0.01
glitch_amplitude = 0.1
glitch_mask = np.random.rand(N) < glitch_prob
glitch = glitch_mask * glitch_amplitude
sig_glitch = sig_ideal + glitch + np.random.randn(N) * base_noise

print(f"[Timing] Data Generation: {time.time() - t_gen:.4f}s")

# --- 3. Timing: Analysis & Plotting ---
t_plot = time.time()

# Analyze and plot results with 3 cases
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
fig.suptitle('Harmonic Decomposition - Thermal Noise vs Nonlinearity vs Glitches', fontsize=14, fontweight='bold')

analyze_decomposition_time(sig_noise, ax=axes[0], title='Thermal Noise Only')
analyze_decomposition_time(sig_nonlin, ax=axes[1], title=f'Nonlinearity (k2={k2:.3f}, k3={k3:.3f})')
analyze_decomposition_time(sig_glitch, ax=axes[2], title=f'Glitches (prob={glitch_prob*100:.2f}%, amp={glitch_amplitude:.1f})')

plt.tight_layout()

print(f"[Timing] Analysis & Plotting: {time.time() - t_plot:.4f}s")

# --- 4. Timing: File Saving (Rendering) ---
t_save = time.time()

fig_path = (output_dir / 'exp_a11_decompose_harmonics.png').resolve()
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"[Save fig] -> [{fig_path}]")
print(f"[Timing] Image Rendering & Saving: {time.time() - t_save:.4f}s")

print(f"--- Total Runtime: {time.time() - t_start:.4f}s ---\n")
