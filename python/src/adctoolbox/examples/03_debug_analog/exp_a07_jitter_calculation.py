"""Jitter calculation: Sweep jitter levels at different frequencies"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, plot_error_hist_phase

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Fixed parameters
N = 2**13
Fs = 10e9
Fin_list = [100e6, 1000e6, 10000e6]
A, DC = 0.49, 0.5 
base_noise = 10e-6

# Jitter sweep: 100 fs to 10 ps (logarithmic spacing)
jitter_levels = np.logspace(-15, -20, 12)

print(f"[Jitter Calculation] [Fs={Fs/1e9:.0f}GHz, N={N}]")
print(f"  Sweep: {jitter_levels[0]*1e15:.0f}fs to {jitter_levels[-1]*1e12:.1f}ps (15 points)")
print(f"  Testing {len(Fin_list)} frequencies: {[f/1e6 for f in Fin_list]} MHz\n")

# Create figure with 3 subplots side by side
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, Fin in enumerate(Fin_list):
    Fin_actual, bin_idx = find_coherent_frequency(Fs, Fin, N)

    print(f"[{idx+1}/{len(Fin_list)}] Fin = {Fin/1e6:.0f} MHz (actual = {Fin_actual/1e6:.3f} MHz)")

    # Arrays to store results
    jitter_set = []
    jitter_measured = []
    sndr_values = []

    for i, jitter_rms in enumerate(jitter_levels):
        # Generate signal with phase jitter
        np.random.seed(42 + i + idx*100)  # Different seed for each point and frequency
        t = np.arange(N) / Fs

        # Phase jitter model: phase_noise_rms = 2*pi*Fin*jitter
        phase_noise_rms = 2 * np.pi * Fin_actual * jitter_rms
        phase_jitter = np.random.randn(N) * phase_noise_rms

        signal = A * np.sin(2*np.pi*Fin_actual*t + phase_jitter) + DC + np.random.randn(N) * base_noise

        # Measure jitter using plot_error_hist_phase
        error_mean, error_rms, phase_bins, amplitude_noise, phase_noise, error, phase = plot_error_hist_phase(
            signal, bins=100, freq=Fin_actual/Fs, disp=0)

        # Convert phase noise back to jitter: Tj = phase_noise / (2*pi*Fin)
        jitter_calc = phase_noise / (2 * np.pi * Fin_actual)

        # Calculate SNDR approximation from phase noise
        # SNDR ≈ -20*log10(phase_noise) for phase-noise limited signals
        sndr_approx = -20 * np.log10(phase_noise) if phase_noise > 0 else 100

        jitter_set.append(jitter_rms)
        jitter_measured.append(jitter_calc)
        sndr_values.append(sndr_approx)

    # Convert to numpy arrays
    jitter_set = np.array(jitter_set)
    jitter_measured = np.array(jitter_measured)
    sndr_values = np.array(sndr_values)

    # Calculate metrics
    correlation = np.corrcoef(jitter_set, jitter_measured)[0, 1]
    errors_pct = np.abs(jitter_measured - jitter_set) / jitter_set * 100
    avg_error = np.mean(errors_pct)

    print(f"  Correlation = {correlation:.4f}, Avg Error = {avg_error:.2f}%\n")

    # Plot: Measured vs Set jitter (left axis) + SNDR (right axis)
    ax1 = axes[idx]
    ax1.loglog(jitter_set*1e15, jitter_set*1e15, 'k--', linewidth=1.5, label='Set jitter')
    ax1.loglog(jitter_set*1e15, jitter_measured*1e15, 'bo', linewidth=2, markersize=8, markerfacecolor='b', label='Calculated jitter')
    ax1.set_xlabel('Set Jitter (fs)', fontsize=12)
    ax1.set_ylabel('Jitter (fs)', fontsize=12, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylim([jitter_set.min()*1e15*0.5, jitter_set.max()*1e15*2])
    ax1.grid(True, which='both', alpha=0.3)

    # Right axis for SNDR
    ax2 = ax1.twinx()
    ax2.semilogx(jitter_set*1e15, sndr_values, 's-', color='red', linewidth=2, markersize=8, label='SNDR')
    ax2.set_ylabel('SNDR (dB)', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim([0, 100])

    # Title
    ax1.set_title(f'Fin = {Fin/1e6:.0f} MHz', fontsize=13, fontweight='bold')

    # Combine legends (only for first subplot)
    if idx == 0:
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=10)

fig.suptitle(f'Jitter Measurement Accuracy (Fs = {Fs/1e9:.0f} GHz)', fontsize=14, fontweight='bold')
plt.tight_layout()
fig_path = output_dir / 'exp_a07_jitter_calculation.png'
plt.savefig(fig_path, dpi=150)
plt.close()

print(f"[Save fig] -> [{fig_path}]")
