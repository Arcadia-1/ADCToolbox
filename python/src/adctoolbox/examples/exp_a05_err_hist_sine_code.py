"""Error histogram vs code value: noise, jitter, harmonic distortion, kickback"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_bin, err_hist_sine

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N = 2**13
Fs = 800e6
Fin_target = 80e6
J = find_bin(Fs, Fin_target, N)
Fin = J * Fs / N
t = np.arange(N) / Fs
A, DC = 0.49, 0.5
base_noise = 50e-6

# Signal 1: Noise
noise_rms = 180e-6
signal_noise = A * np.sin(2*np.pi*Fin*t) + DC + np.random.randn(N) * noise_rms

# Signal 2: Jitter
jitter_rms = 1.3e-12
phase_jitter = np.random.randn(N) * 2 * np.pi * Fin * jitter_rms
signal_jitter = A * np.sin(2*np.pi*Fin*t + phase_jitter) + DC + np.random.randn(N) * base_noise

# Signal 3: Harmonic distortion
hd2_dB, hd3_dB = -80, -73
hd2, hd3 = 10**(hd2_dB/20), 10**(hd3_dB/20)
signal_harmonic = A * np.sin(2*np.pi*Fin*t) + DC + hd2 * np.sin(2*2*np.pi*Fin*t) + hd3 * np.sin(3*2*np.pi*Fin*t) + np.random.randn(N) * base_noise

# Signal 4: Kickback
t_ext = np.arange(N+1) / Fs
sig_clean_ext = A * np.sin(2*np.pi*Fin*t_ext) + DC + np.random.randn(N+1) * base_noise
msb_ext = np.floor(sig_clean_ext * 2**4) / 2**4
lsb_ext = np.floor((sig_clean_ext - msb_ext) * 2**12) / 2**12
msb_shifted = msb_ext[:-1]
msb = msb_ext[1:]
lsb = lsb_ext[1:]
kickback_strength = 0.009
signal_kickback = msb + lsb + kickback_strength * msb_shifted

signals = [signal_noise, signal_jitter, signal_harmonic, signal_kickback]
titles = ['Noise', 'Jitter', 'Harmonic Distortion', 'Kickback']
params = [f'RMS = {noise_rms*1e3:.2f} mV',
          f'{jitter_rms*1e12:.1f} ps',
          f'HD2 = {hd2_dB} dB, HD3 = {hd3_dB} dB',
          f'strength = {kickback_strength}']

print(f"[Error Histogram vs Code] [Fs = {Fs/1e6:.0f} MHz, Fin = {Fin/1e6:.1f} MHz, N = {N}]")

# Create 2x2 figure for comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, (signal, title, param) in enumerate(zip(signals, titles, params)):
    # Use mode=1 for code/value domain
    emean, erms, code_values, anoi, pnoi, err, xx = err_hist_sine(signal, bin=100, mode=1, disp=0)

    # Plot RMS error vs code value
    axes[i].bar(code_values, erms*1e6, width=(code_values[1]-code_values[0])*0.9,
                color='skyblue', edgecolor='none', alpha=0.8)
    axes[i].set_xlabel('Code Value (V)', fontsize=11)
    axes[i].set_ylabel('RMS Error (µV)', fontsize=11)
    axes[i].set_title(f'{title}\n{param}', fontsize=12, fontweight='bold')
    axes[i].grid(True, alpha=0.3)
    axes[i].set_xlim([DC - A*1.05, DC + A*1.05])

    print(f"  {title:20s} - RMS range: [{np.min(erms)*1e6:.2f}, {np.max(erms)*1e6:.2f}] uV")

plt.tight_layout()
fig_path = output_dir / f'exp_a06_err_hist_sine_code_fin_{int(Fin/1e6)}M.png'
plt.savefig(fig_path, dpi=150)
plt.close()

print(f"\n[Save fig] -> [{fig_path}]")
