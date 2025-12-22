"""
Comprehensive ADC Analysis Dashboard: Single thermal noise case with 12 diagnostic plots.

This example demonstrates using the generate_aout_dashboard function
to create a 3x4 panel showing 12 different analysis perspectives of the same signal.
Uses a thermal noise case to show how each analysis tool reveals different aspects
of ADC performance.

12 Analysis Tools:
1. analyze_spectrum - Frequency domain analysis (SNDR, SFDR, ENOB)
2. analyze_spectrum_polar - Polar spectrum view
3. analyze_error_by_value - Error vs. ADC code (INL shape)
4. analyze_error_by_phase - Error vs. input phase
5. analyze_decomposition_time - Time-domain signal decomposition
6. analyze_decomposition_polar - Polar decomposition view
7. analyze_error_pdf - Error probability distribution
8. analyze_error_autocorr - Error autocorrelation
9. analyze_error_spectrum - Error signal spectrum
10. analyze_error_envelope_spectrum - Error envelope spectrum
11. analyze_phase_plane - Phase plane (signal vs derivative)
12. analyze_error_phase_plane - Error phase plane (residual vs amplitude)
"""

import time
from pathlib import Path

# --- 1. Timing: Imports ---
t_start = time.time()

# Set non-interactive backend before importing analysis libraries
import matplotlib
matplotlib.use('Agg')

import numpy as np
from adctoolbox.toolset import generate_aout_dashboard
from adctoolbox import find_coherent_frequency

print(f"[Timing] Library Imports: {time.time() - t_start:.4f}s")

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# --- 2. Timing: Data Generation ---
t_gen = time.time()

# Setup parameters - Thermal noise case
N = 2**16
Fs = 800e6
Fin_target = 10e6
Fin, Fin_bin = find_coherent_frequency(fs=Fs, fin_target=Fin_target, n_fft=N)
t = np.arange(N) / Fs
A = 0.49
DC = 0.5
base_noise = 50e-6
resolution = 12

print(f"[Config] Fs={Fs/1e6:.0f} MHz, Fin={Fin/1e6:.2f} MHz, Bin/N=[{Fin_bin}/{N}], Resolution={resolution} bits")
print(f"[Config] Signal: A={A:.3f} V, DC={DC:.3f} V, Noise RMS={base_noise*1e6:.2f} uVrms")

# Generate ideal ADC with thermal noise
signal = A * np.sin(2 * np.pi * Fin * t) + DC + np.random.randn(N) * base_noise

print(f"[Timing] Data Generation: {time.time() - t_gen:.4f}s")

# --- 3. Timing: Analysis & Plotting & Saving (Dashboard Generation) ---
t_plot = time.time()

fig_path = (output_dir / 'exp_t01_aout_dashboard.png').resolve()

fig, axes = generate_aout_dashboard(
    signal=signal,
    fs=Fs,
    freq=Fin,  # Pass frequency in Hz (not normalized)
    resolution=resolution,
    output_path=fig_path
)

print(f"[Timing] Dashboard Generation & Saving: {time.time() - t_plot:.4f}s")

print(f"\n--- Total Runtime: {time.time() - t_start:.4f}s ---\n")
