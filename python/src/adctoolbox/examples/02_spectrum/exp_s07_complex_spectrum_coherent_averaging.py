"""Coherent spectrum averaging with multiple runs.

Demonstrates how coherent averaging (phase alignment) improves the spectrum
compared to traditional power averaging when signals have random phase offsets.

Uses the analyze_spectrum_coherent_averaging wrapper for clean, consistent interface.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import calc_coherent_freq, analyze_spectrum
from adctoolbox.aout import analyze_spectrum_coherent_averaging

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N_fft = 2**10
Fs = 100e6
Fin, Fin_bin = calc_coherent_freq(fs=Fs, fin_target=5e6, n_fft=N_fft)
print(f"[Coherent Spectrum Averaging Comparison] Fs={Fs/1e6:.1f} MHz, Fin={Fin/1e6:.4f} MHz, Bin={Fin_bin}, N_fft={N_fft}")

# Signal parameters - same as exp_b04
A = 0.499
noise_rms = 100e-6
hd2_dB = -100
hd3_dB = -90
hd2_amp = 10**(hd2_dB/20)
hd3_amp = 10**(hd3_dB/20)
k2 = hd2_amp / (A / 2)
k3 = hd3_amp / (A**2 / 4)

# Number of runs to test
N_runs = [1, 10, 100]

# Generate signals for all runs - same method as exp_b04
t = np.arange(N_fft) / Fs
N_max = max(N_runs)
signal_matrix = np.zeros((N_fft, N_max))


for run_idx in range(N_max):
    phase_random = np.random.uniform(0, 2 * np.pi)
    sig_ideal = A * np.sin(2 * np.pi * Fin * t + phase_random)

    # Apply static nonlinearity: y = x + k2*x^2 + k3*x^3
    sig_distorted = sig_ideal + k2 * sig_ideal**2 + k3 * sig_ideal**3 + np.random.randn(N_fft) * noise_rms

    signal_matrix[:, run_idx] = sig_distorted

print(f"\n[Generated {N_max} runs with random phase, noise, and nonlinearity]")

# Create comparison plots
fig, axes = plt.subplots(2, len(N_runs), figsize=(16, 9))


for idx, N_run in enumerate(N_runs):
    print(f"\n[Processing {N_run:3d} run(s)]")

    # Prepare signal data
    if N_run == 1:
        signal_single = signal_matrix[:, 0]
        signal_data = signal_single
    else:
        signal_data = signal_matrix[:, :N_run].T

    # Traditional power averaging (analyze_spectrum)
    plt.sca(axes[0, idx])
    result_trad = analyze_spectrum(signal_data, fs=Fs)
    axes[0, idx].set_ylim([-120, 0])

    # Coherent averaging (analyze_spectrum_coherent_averaging)
    plt.sca(axes[1, idx])
    result_coh = analyze_spectrum_coherent_averaging(signal_data, fs=Fs)
    axes[1, idx].set_ylim([-120, 0])

# Column titles
for i, N_run in enumerate(N_runs):
    # axes[0, i].set_title(f'Spectrum (N_run = {N_run})', fontsize=14, fontweight='bold')
    axes[1, i].set_title(f'Coherent averaging (N_run = {N_run})', fontsize=14, fontweight='bold')

# Row labels
axes[0, 0].set_ylabel('Power Spectrum (dB)', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Coherent Spectrum (dBFS)', fontsize=11, fontweight='bold')

# Add overall title
fig.suptitle(f'Power Spectrum Averaging vs Complex Spectrim Coherent Averaging (N_fft = {N_fft}, Random Phase Offsets)',
             fontsize=16, fontweight='bold')

plt.tight_layout()
fig_path = output_dir / 'exp_b06_coherent_averaging_comparison.png'
print(f"\n[Save comparison] -> [{fig_path}]")
plt.savefig(fig_path, dpi=150)
plt.close(fig)
