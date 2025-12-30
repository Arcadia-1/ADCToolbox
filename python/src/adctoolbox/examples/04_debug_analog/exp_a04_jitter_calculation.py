"""Jitter calculation: Sweep jitter levels and measure recovery accuracy

Demonstrates jitter measurement using analyze_error_by_phase:
- Generate signals with known jitter levels
- Recover jitter using PM noise from phase error analysis
- Compare measured vs theoretical SNR
- Validate against theoretical jitter limit curve

Similar to MATLAB run_jitter_load.m
"""

import time

# --- 1. Timing: Imports ---
t_start = time.time()
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import analyze_error_by_phase, find_coherent_frequency, analyze_spectrum
from adctoolbox.fundamentals.metrics import calculate_jitter_limit

print(f"[Timing] Library Imports: {time.time() - t_start:.4f}s")

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# --- 2. Setup Parameters ---
t_setup = time.time()

# Fixed parameters
N = 2**16
Fs = 7e9
Fin_target_list = [100e6, 1000e6, 2000e6]
A = 0.49
DC = 0.0
base_noise = 50e-6

# Jitter sweep (logarithmic spacing)
jitter_levels = np.logspace(-15, -11, 30)

print(f"[Config] Fs={Fs/1e9:.0f} GHz, N={N}, A={A:.3f} V, DC={DC:.3f} V")
print(f"[Jitter Sweep] {jitter_levels[0]*1e15:.1f} fs to {jitter_levels[-1]*1e12:.1f} ps ({len(jitter_levels)} points)")
print(f"[Frequencies] Testing {len(Fin_target_list)} frequencies: {[int(f/1e6) for f in Fin_target_list]} MHz\n")

print(f"[Timing] Setup: {time.time() - t_setup:.4f}s")

# --- 3. Analysis & Plotting ---
t_analysis = time.time()

# Create figure with 2 rows x 3 columns
fig, axes = plt.subplots(2, 3, figsize=(20, 10))

# Test with and without thermal noise
noise_levels = [0.0, base_noise]
noise_labels = ['Without Thermal Noise', f'With {base_noise*1e6:.0f} uV Thermal Noise']

for row_idx, (noise_level, noise_label) in enumerate(zip(noise_levels, noise_labels)):
    print(f"\n{'='*80}")
    print(f"Row {row_idx + 1}: {noise_label}")
    print(f"{'='*80}")

    for col_idx, Fin_target in enumerate(Fin_target_list):
        # Find coherent frequency
        Fin, Fin_bin = find_coherent_frequency(fs=Fs, fin_target=Fin_target, n_fft=N)

        print(f"[{col_idx+1}/{len(Fin_target_list)}] Fin_target = {Fin_target/1e6:.0f} MHz, Fin = {Fin/1e6:.3f} MHz, Bin = {Fin_bin}/{N}")

        # Arrays to store results
        jitter_set = []
        jitter_measured = []
        snr_measured = []

        for i, jitter_rms in enumerate(jitter_levels):
            # Generate signal with phase jitter
            t = np.arange(N) / Fs

            # Phase jitter model
            phase_noise_rms = 2 * np.pi * Fin * jitter_rms
            phase_jitter = np.random.randn(N) * phase_noise_rms

            signal = A * np.sin(2*np.pi*Fin*t + phase_jitter) + DC + np.random.randn(N) * noise_level

            # Measure jitter using analyze_error_by_phase
            results = analyze_error_by_phase(signal, norm_freq=Fin/Fs, n_bins=100,
                                            include_base_noise=True, create_plot=False)

            # Extract PM noise and convert to jitter
            pm_noise_rad = results['pm_noise_rms_rad']
            jitter_calc = pm_noise_rad / (2 * np.pi * Fin)

            # Calculate SNR using analyze_spectrum
            spec_results = analyze_spectrum(signal, fs=Fs, osr=1, create_plot=False)
            snr_db = spec_results['snr_dbc']

            jitter_set.append(jitter_rms)
            jitter_measured.append(jitter_calc)
            snr_measured.append(snr_db)

        # Convert to numpy arrays
        jitter_set = np.array(jitter_set)
        jitter_measured = np.array(jitter_measured)
        snr_measured = np.array(snr_measured)

        # Calculate theoretical jitter limit SNR
        snr_theoretical = calculate_jitter_limit(Fin, jitter_set)

        # Calculate metrics
        correlation = np.corrcoef(jitter_set, jitter_measured)[0, 1]
        errors_pct = np.abs(jitter_measured - jitter_set) / jitter_set * 100
        avg_error = np.mean(errors_pct)

        print(f"  Correlation = {correlation:.4f}, Avg Error = {avg_error:.2f}%")

        # Plot: Measured vs Set jitter (left axis) + SNR (right axis)
        ax1 = axes[row_idx, col_idx]
        ax1.loglog(jitter_set*1e15, jitter_set*1e15, 'k--', linewidth=1.5, label='Set jitter')
        ax1.loglog(jitter_set*1e15, jitter_measured*1e15, 'bo', linewidth=2, markersize=8,
                   markerfacecolor='b', label='Calculated jitter')
        ax1.set_xlabel('Set Jitter (fs)', fontsize=12)
        ax1.set_ylabel('Jitter (fs)', fontsize=12, color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_ylim([jitter_set.min()*1e15*0.5, jitter_set.max()*1e15*2])
        ax1.grid(True, which='both', alpha=0.3)

        # Right axis for SNR
        ax2 = ax1.twinx()
        ax2.semilogx(jitter_set*1e15, snr_measured, 's-', color='red', linewidth=2,
                     markersize=8, label='Measured SNR')
        ax2.semilogx(jitter_set*1e15, snr_theoretical, '--', color='darkred', linewidth=2,
                     label='Theoretical SNR (Jitter Limit)')
        ax2.set_ylabel('SNR (dB)', fontsize=12, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim([0, 120])

        # Title with noise information
        ax1.set_title(f'Fin = {Fin/1e6:.0f} MHz ({noise_label})', fontsize=11)

        # Add text annotation with correlation and average error
        text_str = f'Corr = {correlation:.4f}\nAvg Err = {avg_error:.2f}%'
        ax1.text(0.5, 0.02, text_str, transform=ax1.transAxes,
                fontsize=10, verticalalignment='bottom', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Combine legends (show in all subplots)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        legend = ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', fontsize=9, framealpha=1)

print(f"\n[Timing] Analysis & Plotting: {time.time() - t_analysis:.4f}s")

# --- 4. Save Figure ---
t_save = time.time()

fig.suptitle(f'Jitter Recovery Using analyze_error_by_phase - Calculated vs Set Jitter (Fs = {Fs/1e9:.0f} GHz)', fontsize=14, fontweight='bold')
plt.tight_layout()
fig_path = output_dir / 'exp_a04_jitter_calculation.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"[Save fig] -> [{fig_path.resolve()}]")
print(f"[Timing] Image Saving: {time.time() - t_save:.4f}s")

print(f"\n--- Total Runtime: {time.time() - t_start:.4f}s ---\n")
