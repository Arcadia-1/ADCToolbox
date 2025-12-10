"""Harmonic decomposition comparison: Time-domain and Polar (LMS mode) plots"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import (
    find_coherent_frequency,
    compute_harmonic_decomposition,
    plot_harmonic_decomposition_time,
    plot_harmonic_decomposition_polar
)

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Parameters
N = 2**13
Fs = 800e6
Fin_target = 10e6
Fin, Fin_bin = find_coherent_frequency(Fs, Fin_target, N)
A = 0.49

sig_ideal = A * np.sin(2 * np.pi * Fin * np.arange(N) / Fs)
print(f"[Harmonic Decomposition - Time vs Polar] Fs={Fs/1e6:.1f} MHz, Fin={Fin/1e6:.6f} MHz")

# Test case: Static nonlinearity
k2 = 0.001
k3 = 0.005
noise_rms = 50e-6
signal_distorted = sig_ideal + k2 * sig_ideal**2 + k3 * sig_ideal**3 + np.random.randn(N) * noise_rms

# Compute harmonic decomposition
results = compute_harmonic_decomposition(signal_distorted, normalized_freq=Fin/Fs, order=10)
print(f"[Signal] Ideal sine + k2={k2:.3f}*x^2 + k3={k3:.3f}*x^3 + noise={noise_rms*1e6:.0f}uV RMS")

# Create figure with 2 rows, 1 column (time on top, polar on bottom)
fig = plt.figure(figsize=(14, 10))
fig.suptitle(f'Harmonic Decomposition: Time-Domain vs Polar (LMS) Domain\nk2={k2:.3f}, k3={k3:.3f}',
             fontsize=16, fontweight='bold')

# Row 1: Time-domain plot
ax1 = plt.subplot(2, 1, 1)
plot_harmonic_decomposition_time(results, ax=ax1)

# Row 2: Polar plot (LMS mode)
ax2 = plt.subplot(2, 1, 2, projection='polar')
plot_harmonic_decomposition_polar(results, ax=ax2)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig_path = output_dir / 'exp_a24_decompose_harmonics_comparison.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')

print(f"[Time Domain] Shows signal waveform and harmonic/other errors vs samples")
print(f"[Polar Domain] Shows fundamental and harmonics as phasors (phase vs magnitude)")
print(f"\n[Save fig] -> [{fig_path}]\n")
plt.close(fig)
