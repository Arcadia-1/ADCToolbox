"""Harmonic decomposition: thermal noise vs static nonlinearity

This example demonstrates the consolidated ADCToolbox analysis functions:
1. analyze_harmonic_decomposition - Decompose signal into harmonics
2. analyze_error_by_phase - Separate AM/PM error components
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import analyze_harmonic_decomposition

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N = 2**13
Fs = 800e6
Fin = 10.1234567e6
t = np.arange(N) / Fs
A = 0.499
DC = 0.5
sig_ideal = A * np.sin(2 * np.pi * Fin * np.arange(N) / Fs) + DC

# Case 1: Thermal noise only
noise_rms = 500e-6
signal_noise = sig_ideal + np.random.randn(N) * noise_rms
print(f"[Sinewave] [Fs={Fs/1e6:.1f} MHz] [Fin={Fin/1e6:.6f} MHz] [Amplitude={A:.3f} V] [DC={DC:.3f} V] [Noise RMS={noise_rms*1e3:.2f} mV]")

# Case 2: Static nonlinearity (k2 and k3) + base noise
k2 = 0.001
k3 = 0.005
base_noise_rms = 50e-6
signal_nonlin = sig_ideal + k2 * sig_ideal**2 + k3 * sig_ideal**3 + np.random.randn(N) * base_noise_rms
print(f"[Sinewave with Nonlinearity] [k2={k2:.3f}] [k3={k3:.3f}] [Base Noise RMS={base_noise_rms*1e3:.2f} mV]")

# Analyze and plot harmonic decomposition for both cases
analyze_harmonic_decomposition(signal_noise, order=3, show_plot=True)
plt.suptitle(f'Case 1: Harmonic Decomposition - Thermal Noise ({noise_rms*1e6:.0f}uV RMS)', fontsize=14, fontweight='bold')
fig_path = output_dir / 'exp_a21_decompose_harmonics_thermal.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"[Save fig] -> [{fig_path}]")
plt.close()

analyze_harmonic_decomposition(signal_nonlin, order=3, show_plot=True)
plt.suptitle(f'Case 2: Harmonic Decomposition - Static Nonlinearity (k2={k2:.3f}, k3={k3:.3f})', fontsize=14, fontweight='bold')
fig_path = output_dir / 'exp_a21_decompose_harmonics_nonlin.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"[Save fig] -> [{fig_path}]\n")
plt.close()