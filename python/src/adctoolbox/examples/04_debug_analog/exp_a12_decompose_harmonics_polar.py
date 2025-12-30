"""
Demonstrates `analyze_decomposition_polar` for distinguishing thermal noise from static nonlinearity.
This method provides polar visualization of harmonic decomposition to identify nonlinearity
with magnitude and phase information in polar coordinates.
Includes spectrum analysis for comparison.
"""

import time

# --- 1. Timing: Imports ---
t_start = time.time()
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import analyze_decomposition_polar, analyze_spectrum, find_coherent_frequency

print(f"[Timing] Library Imports: {time.time() - t_start:.4f}s")

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# --- 2. Timing: Data Generation ---
t_gen = time.time()

# Setup parameters with coherent frequency
N = 2**13
Fs = 800e6
Fin_target = 100e6
Fin, Fin_bin = find_coherent_frequency(fs=Fs, fin_target=Fin_target, n_fft=N)
t = np.arange(N) / Fs
A = 0.5
DC = 0.5
base_noise = 50e-6
adc_range = [0, 1]  # ADC FSR for spectrum analysis

sig_ac = A * np.sin(2 * np.pi * Fin * t)  # AC component only
sig_ideal = sig_ac + DC
print(f"[Config] Fs={Fs/1e6:.0f} MHz, Fin={Fin/1e6:.6f} MHz, Bin={Fin_bin}, N={N}")

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

# Create 2x3 subplot grid: first row spectrum, second row polar
fig = plt.figure(figsize=(18, 12))
fig.suptitle('Harmonic Decomposition - Thermal Noise vs Nonlinearity vs Glitches', fontsize=14, fontweight='bold')

# First row: Spectrum plots
ax_spec1 = plt.subplot(2, 3, 1)
ax_spec2 = plt.subplot(2, 3, 2)
ax_spec3 = plt.subplot(2, 3, 3)

# Second row: Polar plots
ax_polar1 = plt.subplot(2, 3, 4, projection='polar')
ax_polar2 = plt.subplot(2, 3, 5, projection='polar')
ax_polar3 = plt.subplot(2, 3, 6, projection='polar')

# Plot spectrums
plt.sca(ax_spec1)
result_noise = analyze_spectrum(sig_noise, fs=Fs, max_scale_range=adc_range)
ax_spec1.set_title(f'Thermal Noise Only\nSFDR = {result_noise["sfdr_dbc"]:.2f} dB, SNDR = {result_noise["sndr_dbc"]:.2f} dB', fontsize=10)

plt.sca(ax_spec2)
result_nonlin = analyze_spectrum(sig_nonlin, fs=Fs, max_scale_range=adc_range)
ax_spec2.set_title(f'Nonlinearity (k2={k2:.3f}, k3={k3:.3f})\nSFDR = {result_nonlin["sfdr_dbc"]:.2f} dB, SNDR = {result_nonlin["sndr_dbc"]:.2f} dB', fontsize=10)

plt.sca(ax_spec3)
result_glitch = analyze_spectrum(sig_glitch, fs=Fs, max_scale_range=adc_range)
ax_spec3.set_title(f'Glitches (prob={glitch_prob*100:.2f}%, amp={glitch_amplitude:.1f})\nSFDR = {result_glitch["sfdr_dbc"]:.2f} dB, SNDR = {result_glitch["sndr_dbc"]:.2f} dB', fontsize=10)

# Plot polar decompositions
analyze_decomposition_polar(sig_noise, ax=ax_polar1, title='Thermal Noise Only')
analyze_decomposition_polar(sig_nonlin, ax=ax_polar2, title=f'Nonlinearity (k2={k2:.3f}, k3={k3:.3f})')
analyze_decomposition_polar(sig_glitch, ax=ax_polar3, title=f'Glitches (prob={glitch_prob*100:.2f}%, amp={glitch_amplitude:.1f})')

plt.tight_layout()

print(f"[Timing] Analysis & Plotting: {time.time() - t_plot:.4f}s")

# --- 4. Timing: File Saving (Rendering) ---
t_save = time.time()

fig_path = (output_dir / 'exp_a12_decompose_harmonics_polar.png').resolve()
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"[Save fig] -> [{fig_path}]")
print(f"[Timing] Image Rendering & Saving: {time.time() - t_save:.4f}s")

print(f"--- Total Runtime: {time.time() - t_start:.4f}s ---\n")
