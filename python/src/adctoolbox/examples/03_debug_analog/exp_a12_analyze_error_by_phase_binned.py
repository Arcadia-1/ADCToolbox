"""Phase-based error analysis: Binned approach for trend analysis

This example demonstrates the analyze_error_by_phase wrapper function using
binned mode for robust trend analysis of phase-dependent noise (AM/PM decomposition).

Compares two cases:
1. Ideal ADC with thermal noise (50 uVrms)
2. ADC with phase jitter (0.05 rad RMS)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import analyze_error_by_phase

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Setup parameters
N = 2**16
Fs = 800e6
Fin = 10.567e6  # no need to be coherent for phase error analysis
normalized_freq = Fin / Fs
t = np.arange(N) / Fs
A = 0.49
num_bits = 10
base_noise = 50e-6
print(f"[Config] Fs={Fs/1e6:.0f} MHz, Fin={Fin/1e6:.2f} MHz, N={N}, Bits={num_bits}")

# Case 1: Ideal ADC with Thermal Noise
sig_noise = A * np.sin(2 * np.pi * Fin * t) + np.random.randn(N) * base_noise

# Case 2: ADC with Phase Jitter
phase_jitter = 0.05 * np.random.randn(N)  # 0.05 rad RMS
jitter_phase = 2 * np.pi * Fin * t + phase_jitter
sig_jitter = A * np.sin(jitter_phase) + np.random.randn(N) * base_noise

# Analyze and plot results - Binned mode (with binned trend plot)
print("\n[Analyzing Case 1: Thermal Noise (Binned mode)]")
analyze_error_by_phase(sig_noise, normalized_freq, mode="binned", bin_count=100, show_plot=True, plot_mode="binned")
fig_path = (output_dir / 'exp_a12_analyze_error_by_phase_thermal_noise.png').resolve()
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"[Save fig] -> [{fig_path}]\n")
plt.close()

print("[Analyzing Case 2: Phase Jitter (Binned mode)]")
analyze_error_by_phase(sig_jitter, normalized_freq, mode="binned", bin_count=100, show_plot=True, plot_mode="binned")
fig_path = (output_dir / 'exp_a12_analyze_error_by_phase_jitter.png').resolve()
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"[Save fig] -> [{fig_path}]\n")
plt.close()
