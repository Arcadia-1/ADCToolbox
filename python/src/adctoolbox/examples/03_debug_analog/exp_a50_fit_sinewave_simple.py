"""Simple Sinewave Fitting - 1st Order with Noise

This example demonstrates the basic fit_sine_harmonics function
for fitting a single sinewave (1st order) in the presence of noise.

Focus: Basic noise robustness and fitting accuracy
Visualization: 2 plots (top: fitted vs raw, bottom: residual)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, amplitudes_to_snr, snr_to_nsd
from adctoolbox.aout import fit_sine_harmonics

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Setup parameters
N = 2**13
Fs = 800e6
Fin_target = 10e6
Fin, Fin_bin = find_coherent_frequency(Fs, Fin_target, N)
normalized_freq = Fin / Fs
t = np.arange(N) / Fs
A = 0.49
noise_rms = 20e-3

snr_fundamental = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=noise_rms)
nsd_fundamental = snr_to_nsd(snr_fundamental, fs=Fs, osr=1)
print(f"[Sinewave] Fs={Fs/1e6:.1f} MHz, Fin={Fin/1e6:.6f} MHz, A={A:.3f} V, N={N}")
print(f"[Nonideal] Noise RMS=[{noise_rms*1e6:.2f} uVrms], Theoretical SNR=[{snr_fundamental:.2f} dB], Theoretical NSD=[{nsd_fundamental:.2f} dBFS/Hz]")

sig_ideal = A * np.sin(2 * np.pi * Fin * t)

sig_noisy = sig_ideal + np.random.randn(N) * noise_rms

W, sig_fit, A_matrix, phase = fit_sine_harmonics(
    sig_noisy,
    freq=normalized_freq,
    order=1,
    include_dc=True
)

# Extract fitted parameters
dc_fit = W[0]
cos_amp = W[1]
sin_amp = W[2]
mag_fit = np.sqrt(cos_amp**2 + sin_amp**2)
phase_fit = np.arctan2(sin_amp, cos_amp) * 180 / np.pi
freq_fit = Fin

# Calculate errors
residual = sig_noisy - sig_fit
residual_rms = np.sqrt(np.mean(residual**2))
reconstruction_error = np.sqrt(np.mean((sig_ideal - sig_fit)**2))

# Parameter errors
dc_error = abs(0.0 - dc_fit)
mag_error = abs(A - mag_fit)
phase_error = 90.0 - phase_fit  # Expected phase is 90 degrees for sin()
freq_error = abs(Fin - freq_fit)

print(f"[Expected] DC=[{0.0:8.4f} V], Amp=[{A:8.4f} V], Freq=[{Fin/1e6:8.4f} MHz], Phase=[90.00°]")
print(f"[Fitted  ] DC=[{dc_fit:8.4f} V], Amp=[{mag_fit:8.4f} V], Freq=[{freq_fit/1e6:8.4f} MHz], Phase=[{phase_fit:5.2f}°]")
print(f"[Error   ] DC=[{dc_error:8.4f} V], Amp=[{100*mag_error/A:8.4f} %], Freq=[{freq_error/1e6:8.4f} MHz], Phase=[{phase_error:5.2f}°]")

print(f"\nReconstruction error:     {reconstruction_error:.6e} V")
print(f"Residual error (RMS):     {residual_rms:.6e} V")
print(f"Input noise RMS:          {noise_rms:.6e} V")

# Calculate number of samples for 3 cycles
period_samples = Fs / Fin
num_cycles = 3
num_samples = int(period_samples * num_cycles)
sample_range = slice(0, num_samples)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle('Simple Sinewave Fitting (1st Order) with Noise', fontsize=16, fontweight='bold')

# Plot 1: Fitted vs Raw Signal (3 cycles)
axes[0].scatter(t[sample_range]*1e9, sig_noisy[sample_range], c='blue', s=8, alpha=0.5, label='Raw (noisy)')
axes[0].plot(t[sample_range]*1e9, sig_fit[sample_range], 'r-', label='Fitted', linewidth=2)
axes[0].set_title(f'Fitted Signal vs Raw Data ({num_cycles} cycles)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Time (ns)')
axes[0].set_ylabel('Amplitude (V)')
axes[0].legend(fontsize=11, loc='upper right')
axes[0].grid(True, alpha=0.3)

# Plot 2: Residual Error
axes[1].plot(t[sample_range]*1e9, residual[sample_range], 'g-', linewidth=1.2)
axes[1].axhline(y=noise_rms, color='r', linestyle='--', linewidth=1.5, label=f'Input noise RMS = {noise_rms*1e6:.0f} μV')
axes[1].axhline(y=-noise_rms, color='r', linestyle='--', linewidth=1.5)
axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
axes[1].set_title(f'Residual Error (RMS = {residual_rms:.2e} V)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Time (ns)')
axes[1].set_ylabel('Residual (V)')
axes[1].legend(fontsize=11, loc='upper right')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig_path = output_dir / 'exp_a50_fit_sinewave_simple.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"[Save fig] -> [{fig_path}]")
plt.close(fig)
