"""Manual spectrum analysis example using modular calculation and plotting."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import calc_coherent_freq, calculate_spectrum_metrics, plot_spectrum

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N_fft = 2**13
Fs = 100e6
Fin_target = 12e6
Fin, Fin_bin = calc_coherent_freq(fs=Fs, fin_target=Fin_target, n_fft=N_fft)
A = 0.5
noise_rms = 200e-6

print(f"[Analysis Parameters] N = {N_fft}, Fs = {Fs/1e6:.2f} MHz, Fin = {Fin/1e6:.4f} MHz (Bin = {Fin_bin})")

t = np.arange(N_fft) / Fs
signal = A * np.sin(2*np.pi*Fin*t) + np.random.randn(N_fft) * noise_rms

# Step 1: Calculate spectrum metrics (pure computation)
metrics, plot_data = calculate_spectrum_metrics(signal, fs=Fs)

# Step 2: Display results
print(f"[ENOB] = {metrics['enob']:.2f} b, [SNDR] = {metrics['sndr_db']:.2f} dB, [SFDR] = {metrics['sfdr_db']:.2f} dB, [SNR] = {metrics['snr_db']:.2f} dB")

# Step 3: Plot the spectrum (pure visualization)
fig, ax = plt.subplots(figsize=(8, 6))
plot_spectrum(metrics, plot_data, ax=ax)

# Step 4: Save figure
fig_path = (output_dir / 'exp_b04_manual_spectrum.png').resolve()
print(f"\n[Save fig] -> [{fig_path}]\n")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()