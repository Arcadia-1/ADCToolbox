"""Jitter calculation: Sweep jitter levels and verify measurement accuracy"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_bin, err_hist_sine

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Fixed parameters
N = 2**13
Fs = 10e9
Fin = 1000e6  # 1 GHz input frequency
bin_idx = find_bin(Fs, Fin, N)
Fin_actual = bin_idx * Fs / N
A, DC = 0.49, 0.5
base_noise = 10e-6

# Jitter sweep: 100 fs to 10 ps (logarithmic spacing)
jitter_levels = np.logspace(-13, -11, 15)  # 100 fs to 10 ps

print(f"[Jitter Calculation] [Fs={Fs/1e9:.0f}GHz, Fin={Fin/1e6:.0f}MHz, N={N}]")
print(f"  Sweep: {jitter_levels[0]*1e15:.0f}fs to {jitter_levels[-1]*1e12:.1f}ps (15 points)")

# Arrays to store results
jitter_set = []
jitter_measured = []
sndr_values = []

for i, jitter_rms in enumerate(jitter_levels):
    # Generate signal with phase jitter
    np.random.seed(42 + i)  # Different seed for each point
    t = np.arange(N) / Fs

    # Phase jitter model: phase_noise_rms = 2*pi*Fin*jitter
    phase_noise_rms = 2 * np.pi * Fin_actual * jitter_rms
    phase_jitter = np.random.randn(N) * phase_noise_rms

    signal = A * np.sin(2*np.pi*Fin_actual*t + phase_jitter) + DC + np.random.randn(N) * base_noise

    # Measure jitter using err_hist_sine
    emean, erms, phase_deg, anoi, pnoi, err, xx = err_hist_sine(signal, bin=100, fin=Fin_actual/Fs, mode=0, disp=0)

    # Convert phase noise back to jitter: Tj = pnoi / (2*pi*Fin)
    jitter_calc = pnoi / (2 * np.pi * Fin_actual)

    # Calculate SNDR approximation from phase noise
    # SNDR ≈ -20*log10(pnoi) for phase-noise limited signals
    sndr_approx = -20 * np.log10(pnoi) if pnoi > 0 else 100

    jitter_set.append(jitter_rms)
    jitter_measured.append(jitter_calc)
    sndr_values.append(sndr_approx)

    err_pct = abs(jitter_calc-jitter_rms)/jitter_rms*100
    print(f"  [{i+1:2d}/15] [Set={jitter_rms*1e15:6.1f}fs] [Meas={jitter_calc*1e15:6.1f}fs] [Err={err_pct:4.1f}%] [SNDR={sndr_approx:5.1f}dB]")

# Convert to numpy arrays
jitter_set = np.array(jitter_set)
jitter_measured = np.array(jitter_measured)
sndr_values = np.array(sndr_values)

# Calculate metrics
correlation = np.corrcoef(jitter_set, jitter_measured)[0, 1]
errors_pct = np.abs(jitter_measured - jitter_set) / jitter_set * 100
avg_error = np.mean(errors_pct)

status = "PASS" if (correlation > 0.999 and avg_error < 5.0 and np.max(errors_pct) < 10.0) else "FAIL"
print(f"  [Summary] [Corr={correlation:.4f}] [AvgErr={avg_error:.2f}%] [MaxErr={np.max(errors_pct):.2f}%] [{status}]")

# Create figure with 2 subplots
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Top: Measured vs Set jitter
axes[0].loglog(jitter_set*1e15, jitter_measured*1e15, 'bo-', linewidth=2, markersize=8, label='Measured')
axes[0].loglog(jitter_set*1e15, jitter_set*1e15, 'r--', linewidth=2, label='Ideal (y=x)')
axes[0].set_xlabel('Set Jitter (fs)', fontsize=12)
axes[0].set_ylabel('Measured Jitter (fs)', fontsize=12)
axes[0].set_title(f'Jitter Measurement Accuracy (Fin = {Fin/1e6:.0f} MHz)\nCorrelation = {correlation:.4f}, Avg Error = {avg_error:.2f}%',
                  fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, which='both', alpha=0.3)

# Bottom: Error percentage
axes[1].semilogx(jitter_set*1e15, errors_pct, 'ro-', linewidth=2, markersize=8)
axes[1].axhline(0, color='k', linestyle='--', alpha=0.5)
axes[1].axhline(10, color='g', linestyle=':', alpha=0.5, label='±10% target')
axes[1].axhline(-10, color='g', linestyle=':', alpha=0.5)
axes[1].set_xlabel('Set Jitter (fs)', fontsize=12)
axes[1].set_ylabel('Measurement Error (%)', fontsize=12)
axes[1].set_title('Relative Error vs Jitter Level', fontsize=13)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([-50, 50])

plt.tight_layout()
fig_path = output_dir / f'exp_a05_jitter_calculation_fin_{int(Fin/1e6)}M.png'
plt.savefig(fig_path, dpi=150)
plt.close()

print(f"\n[Save fig] -> [{fig_path}]")
