"""Sine wave fitting"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_bin, sine_fit, find_fin

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N = 2**13
Fs = 1e6
Fin = 10e3
J = find_bin(Fs, Fin, N)
Fin = J / N * Fs
t = np.arange(N) / Fs
A = 0.49
DC = 0.5
noise_rms = 20e-3

signal = A * np.sin(2*np.pi*Fin*t) + DC + np.random.randn(N) * noise_rms

data_fit, freq_fit, mag, dc, phi = sine_fit(signal)
residual = signal - data_fit
residual_rms = np.std(residual)

# You can also use find_fin to directly get the fitted frequency
Fin_relative = find_fin(signal)
Fin_abs = find_fin(signal, fs=Fs)

print(f"[real signal] [N = {N:d}] [Fs = {Fs/1e6:.1f} MHz] [Fin = {Fin/1e3:.6f} kHz] [A = {A:.3f}] [DC = {DC:.3f}] [Noise RMS = {noise_rms*1e3:.2f} mV]")
print(f"[sine_fit   ] [Fin/Fs = {freq_fit:.8f}] [Mag = {mag:.3f}] [DC = {dc:.3f}] [Phase = {phi:.3f} rad] [Residual RMS = {residual_rms*1e3:.2f} mV]")
print(f"[find_fin   ] [Fin/Fs = {Fin_relative:.8f}] [Fin = {Fin_abs/1e3:.6f} kHz]")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

freq_fit_abs = freq_fit * Fs
period = 1 / freq_fit_abs
n_periods = 3
n_zoom = int(n_periods * period * Fs)
ax1.plot(t[:n_zoom]*1e6, signal[:n_zoom], 'o', markersize=4, alpha=0.5, label='Raw data')
ax1.plot(t[:n_zoom]*1e6, data_fit[:n_zoom], 'r-', linewidth=2, label='Fitted')
ax1.set_xlabel('Time (us)')
ax1.set_ylabel('Amplitude')
ax1.set_title('Sine Fit: First 3 Periods')
ax1.legend()
ax1.grid(True)

ax2.plot(residual, linewidth=0.5)
ax2.set_xlabel('Sample Index')
ax2.set_ylabel('Residual')
ax2.set_title(f'Residual Error (RMS={residual_rms*1e3:.1f}mV)')
ax2.grid(True)

plt.tight_layout()
fig_path = (output_dir / 'exp_b03_sine_fit.png').resolve()
print(f"[Save fig] -> [{fig_path}]\n")
plt.savefig(fig_path, dpi=150)
plt.close()
