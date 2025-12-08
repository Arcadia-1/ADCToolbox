"""
Basic demo: Spectrum analysis with OSR sweep.

This script demonstrates the effect of oversampling ratio (OSR) on spectrum analysis
by sweeping through different OSR values and plotting the results.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import analyze_spectrum, calc_coherent_freq

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N_fft = 2**16
Fs = 100e6
Fin_target = 0.1e6
Fin, Fin_bin = calc_coherent_freq(fs=Fs, fin_target=Fin_target, n_fft=N_fft)
A = 0.5
noise_rms = 100e-6

print(f"[OSR Sweep Analysis] N = {N_fft}, Fs = {Fs/1e6:.2f} MHz, Fin = {Fin/1e6:.4f} MHz (Bin = {Fin_bin})\n")

t = np.arange(N_fft) / Fs
signal = A * np.sin(2*np.pi*Fin*t) + np.random.randn(N_fft) * noise_rms

# OSR sweep values
osr_values = [1, 2, 4, 8, 10, 100]

# Calculate grid dimensions
n_plots = len(osr_values)
n_cols = 3
n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division

# Create subplot grid
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
if n_plots == 1:
    axes = np.array([axes])
else:
    axes = axes.flatten()

# Store results to calculate SNR improvement
results = []
snr_baseline = None

for idx, osr in enumerate(osr_values):

    plt.sca(axes[idx])
    result = analyze_spectrum(signal, fs=Fs, osr=osr, is_plot=1)
    results.append(result)


    # Store baseline SNR (OSR=1)
    if idx == 0:
        snr_baseline = result['snr_db']
        axes[idx].set_title(f'OSR = {osr}', fontsize=12, fontweight='bold')
        print(f"[OSR = {osr:3d}] ENOB = {result['enob']:5.2f} b, SNR = {result['snr_db']:6.2f} dB")
    else:
        snr_improvement = result['snr_db'] - snr_baseline
        axes[idx].set_title(f'OSR = {osr} (SNR +{snr_improvement:.1f} dB)', fontsize=12, fontweight='bold')
        print(f"[OSR = {osr:3d}] ENOB = {result['enob']:5.2f} b, SNR = {result['snr_db']:6.2f} dB, Delta SNR = {snr_improvement:5.2f} dB")

# Remove empty subplots
for idx in range(n_plots, len(axes)):
    axes[idx].remove()

plt.tight_layout()
fig_path = (output_dir / 'exp_s03_analyze_spectrum_osr.png').resolve()
print(f"\n[Save fig] -> [{fig_path}]\n")
plt.savefig(fig_path, dpi=150)
plt.close()