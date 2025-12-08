"""Sine wave fitting"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import calc_coherent_freq, fit_sine, estimate_frequency

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N = 2**13
Fs = 1e6
Fin_target = 10e3
Fin, J = calc_coherent_freq(Fs, Fin_target, N)
t = np.arange(N) / Fs
A = 0.49
DC = 0.5
noise_rms = 20e-3

signal = A * np.sin(2*np.pi*Fin*t) + DC + np.random.randn(N) * noise_rms
print(f"[real signal] [Fin/Fs = {Fin/Fs:.8f}] [amplitude = {A:.3f}] [DC = {DC:.3f}] [Noise RMS = {noise_rms*1e3:.2f} mV]")

result = fit_sine(signal)
print(f"[fit_sine   ] [Fin/Fs = {result['frequency']:.8f}] [amplitude = {result['amplitude']:.3f}] [DC = {result['dc_offset']:.3f}] [Phase = {result['phase']:.3f} rad] [Residual RMS = {result['rmse']*1e3:.2f} mV]")

# You can also use estimate_frequency to directly get the fitted frequency
Fin_relative = estimate_frequency(signal)
Fin_abs = estimate_frequency(signal, fs=Fs)
print(f"\n[estimate_frequency] [Fin/Fs = {Fin_relative:.8f}] [Fin = {Fin_abs/1e3:.6f} kHz]")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

period = 1 / (result['frequency'] * Fs)
n_periods = 3
n_zoom = int(n_periods * period * Fs)
ax1.plot(t[:n_zoom]*1e6, signal[:n_zoom], 'o', markersize=4, alpha=0.5, label='Raw data')
ax1.plot(t[:n_zoom]*1e6, result['fitted_signal'][:n_zoom], 'r-', linewidth=2, label='Fitted')
ax1.set_xlabel('Time (us)', fontsize=12)
ax1.set_ylabel('Amplitude', fontsize=12)
ax1.set_title(f'Sine Fit: 3 Periods (Noise={noise_rms*1e3:.2f}mV)', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

ax2.plot(result['residuals'], linewidth=0.5)
ax2.set_xlabel('Sample Index', fontsize=12)
ax2.set_ylabel('Residual', fontsize=12)
ax2.set_title(f'Residual (RMS={result["rmse"]*1e3:.2f}mV)', fontsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = (output_dir / 'exp_b06_fit_sine.png').resolve()
print(f"\n[Save fig] -> [{fig_path}]\n")
plt.savefig(fig_path, dpi=150)
plt.close(fig)
