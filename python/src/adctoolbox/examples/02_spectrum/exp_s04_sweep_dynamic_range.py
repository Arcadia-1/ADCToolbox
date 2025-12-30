"""
Dynamic Range Sweep: Spectrum Analysis Across Signal Amplitudes

This example demonstrates how spectrum analysis metrics (ENOB, SNR, SFDR, THD)
vary with signal amplitude, revealing the dynamic range of the measurement system.

Key Concepts:
1. Dynamic Range: The ratio between the largest and smallest measurable signals
2. SNR vs Amplitude: SNR improves with signal amplitude until limited by:
   - Quantization noise (for ideal ADCs)
   - Harmonic distortion (for real ADCs)
   - Clipping (when signal exceeds full scale)

Expected Results:
- Low amplitudes: Quantization noise dominates, low SNR
- Mid amplitudes: SNR increases linearly with amplitude (20 dB/decade)
- High amplitudes: Distortion or clipping limits performance
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, analyze_spectrum, amplitudes_to_snr, snr_to_nsd

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Common parameters
N_fft = 2**13
Fs = 100e6
Fin_target = 12e6
Fin, Fin_bin = find_coherent_frequency(fs=Fs, fin_target=Fin_target, n_fft=N_fft)

# Fixed noise floor (simulates quantization noise or thermal noise)
noise_rms = 100e-6  # 100 uVrms

# Sweep amplitude from -80 dBFS to 0 dBFS to capture SNR=0dB crossover point
# Full scale = 0.5 Vpeak (range: -0.5 to +0.5), so dBFS = 20*log10(A/0.5)
amplitudes_dbfs = np.linspace(-80, 0, 50)
amplitudes = 0.5 * 10**(amplitudes_dbfs / 20)

print(f"[Sinewave] Fs=[{Fs/1e6:.2f} MHz], Fin=[{Fin/1e6:.6f} MHz] (coherent, Bin {Fin_bin}), N=[{N_fft}]")
print(f"[Nonideal] Noise RMS=[{noise_rms*1e6:.2f} uVrms] (fixed)")
print(f"[Sweep] Amplitude: {amplitudes_dbfs[0]:.1f} to {amplitudes_dbfs[-1]:.1f} dBFS ({len(amplitudes)} steps)\n")

print("=" * 80)
print("DYNAMIC RANGE SWEEP")
print("=" * 80)

# Generate signals and analyze
t = np.arange(N_fft) / Fs

# Lists to store results
snr_list = []
noise_floor_list = []
snr_theory_list = []

for A, A_dbfs in zip(amplitudes, amplitudes_dbfs):
    # Generate signal with fixed noise
    signal = A * np.sin(2*np.pi*Fin*t) + np.random.randn(N_fft) * noise_rms

    # Analyze spectrum without plotting (create_plot=False)
    # max_scale_range=[-0.5, 0.5] means full scale range is -0.5V to +0.5V
    result = analyze_spectrum(signal, fs=Fs, max_scale_range=[-0.5, 0.5], create_plot=False)

    # Store results
    snr_list.append(result['snr_dbc'])
    noise_floor_list.append(result['noise_floor_dbfs'])

    # Theoretical SNR for this amplitude
    # Use amplitudes_to_snr utility (handles peak-to-RMS conversion internally)
    snr_theory = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=noise_rms)
    snr_theory_list.append(snr_theory)

    print(f"[A={A_dbfs:6.1f} dBFS] SNR=[{result['snr_dbc']:6.2f} dB] (Theory: {snr_theory:.1f} dB), Noise Floor=[{result['noise_floor_dbfs']:7.2f} dBFS]")

# Convert lists to numpy arrays for plotting
snr_array = np.array(snr_list)
noise_floor_array = np.array(noise_floor_list)
snr_theory_array = np.array(snr_theory_list)

# Find SNR=0dB crossover point for dynamic range marking
if snr_array[0] < 0 and snr_array[-1] > 0:
    amp_at_snr_0 = np.interp(0, snr_array, amplitudes_dbfs)
    amp_max = amplitudes_dbfs[-1]
    dynamic_range_calc = amp_max - amp_at_snr_0
else:
    amp_at_snr_0 = None
    dynamic_range_calc = None

# Create single plot with dual y-axes
fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

# Left y-axis: SNR
color_snr = 'tab:blue'
ax1.set_xlabel('Amplitude (dBFS)', fontsize=12)
ax1.set_ylabel('SNR (dB)', fontsize=12, color=color_snr)
ax1.plot(amplitudes_dbfs, snr_theory_array, '--', linewidth=1.5, alpha=0.7, label='Theoretical SNR', color='black', zorder=1)
ax1.plot(amplitudes_dbfs, snr_array, 'o', linewidth=2, markersize=6, label='Measured SNR', color=color_snr, zorder=2)
ax1.tick_params(axis='y', labelcolor=color_snr)
ax1.grid(True, alpha=0.3)

# Mark dynamic range on plot
if amp_at_snr_0 is not None:
    # Draw horizontal line at SNR = 0 dB
    ax1.axhline(y=0, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
    # Draw vertical lines at dynamic range boundaries
    ax1.axvline(x=amp_at_snr_0, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label=f'DR = {dynamic_range_calc:.1f} dB')
    ax1.axvline(x=amp_max, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
    # Add annotation
    ax1.annotate(f'DR = {dynamic_range_calc:.1f} dB',
                xy=((amp_at_snr_0 + amp_max)/2, 0),
                xytext=((amp_at_snr_0 + amp_max)/2, 10),
                ha='center', fontsize=10, color='red', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red', alpha=0.8))

ax1.legend(loc='upper left')

# Right y-axis: Noise Floor
ax2 = ax1.twinx()
color_nf = 'tab:green'
ax2.set_ylabel('Noise Floor (dBFS)', fontsize=12, color=color_nf)
ax2.plot(amplitudes_dbfs, noise_floor_array, 's-', linewidth=2, markersize=2, label='Noise Floor', color=color_nf)
ax2.tick_params(axis='y', labelcolor=color_nf)

# Set y-axis range: 50 dB wide with noise floor at 1/3 from bottom
# Noise floor is around -71 dBFS, so position it at 1/3 * 50 = 16.67 dB from bottom
avg_noise_floor = np.mean(noise_floor_array)
y_range = 50  # dB
bottom = avg_noise_floor - y_range / 3
top = bottom + y_range
ax2.set_ylim([bottom, top])

ax2.legend(loc='lower right')

fig.suptitle(f'SNR and Noise Floor vs Amplitude (N_fft = {N_fft})',
             fontsize=14, fontweight='bold')
plt.tight_layout()

fig_path = output_dir / 'exp_s04_sweep_dynamic_range.png'
print(f"\n[Save fig] -> [{fig_path}]\n")
plt.savefig(fig_path, dpi=150)
plt.close()

# ============================================================================
# Summary: Dynamic Range Analysis
# ============================================================================
print("=" * 80)
print("SUMMARY: Dynamic Range Analysis")
print("=" * 80)

# Calculate Dynamic Range: from SNR = 0 dB crossover point to maximum amplitude
# Dynamic Range = amplitude at max SNR - amplitude at SNR = 0 dB
# Interpolate to find amplitude where SNR = 0 dB
if snr_array[0] < 0 and snr_array[-1] > 0:
    # Find crossover point using linear interpolation
    amp_at_snr_0 = np.interp(0, snr_array, amplitudes_dbfs)
    amp_max = amplitudes_dbfs[-1]
    dynamic_range = amp_max - amp_at_snr_0
    print(f"Dynamic Range: {amp_at_snr_0:.1f} dBFS (SNR=0dB) to {amp_max:.1f} dBFS (max) = {dynamic_range:.1f} dB")
elif snr_array[0] >= 0:
    # Minimum SNR is already above 0 dB, use first point
    amp_at_snr_0 = amplitudes_dbfs[0]
    amp_max = amplitudes_dbfs[-1]
    dynamic_range = amp_max - amp_at_snr_0
    print(f"Dynamic Range: >{dynamic_range:.1f} dB (minimum SNR = {snr_array[0]:.1f} dB at {amp_at_snr_0:.1f} dBFS)")
else:
    print("Maximum SNR is below 0 dB - insufficient dynamic range")
