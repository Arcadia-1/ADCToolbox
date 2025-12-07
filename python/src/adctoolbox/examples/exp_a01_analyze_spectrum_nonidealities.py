"""Spectrum comparison: noise, jitter, harmonic distortion, kickback"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import calc_coherent_freq, analyze_spectrum

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N = 2**13
Fs = 800e6
Fin_target = 80e6
Fin, J = calc_coherent_freq(Fs, Fin_target, N)
t = np.arange(N) / Fs
A, DC = 0.49, 0.5
base_noise = 50e-6

# Signal 1: Noise
noise_rms = 180e-6
signal_noise = A * np.sin(2*np.pi*Fin*t) + DC + np.random.randn(N) * noise_rms

# Signal 2: Jitter
jitter_rms = 1000e-15
phase_jitter = np.random.randn(N) * 2 * np.pi * Fin * jitter_rms
signal_jitter = A * np.sin(2*np.pi*Fin*t + phase_jitter) + DC + np.random.randn(N) * base_noise

# Signal 3: Harmonic distortion (via static nonlinearity)
hd2_dB, hd3_dB = -80, -66
hd2_amp = 10**(hd2_dB/20)  # Harmonic amplitude / Fundamental amplitude
hd3_amp = 10**(hd3_dB/20)

# Compute nonlinearity coefficients to achieve target HD levels
# HD2: k2 * A^2 / 2 = hd2_amp * A  →  k2 = hd2_amp / (A/2)
# HD3: k3 * A^3 / 4 = hd3_amp * A  →  k3 = hd3_amp / (A^2/4)
k2 = hd2_amp / (A / 2)
k3 = hd3_amp / (A**2 / 4)

# Generate distorted signal: y = x + k2*x^2 + k3*x^3
sinewave = A * np.sin(2*np.pi*Fin*t)
signal_harmonic = sinewave + k2 * sinewave**2 + k3 * sinewave**3 + DC + np.random.randn(N) * base_noise

# Signal 4: Kickback
kickback_strength = 0.009
t_ext = np.arange(N+1) / Fs  # Generate N+1 samples
sig_clean_ext = A * np.sin(2*np.pi*Fin*t_ext) + DC + np.random.randn(N+1) * base_noise
msb_ext = np.floor(sig_clean_ext * 2**4) / 2**4
lsb_ext = np.floor((sig_clean_ext - msb_ext) * 2**12) / 2**12
msb_shifted = msb_ext[:-1]  # First N samples (delayed MSB)
msb = msb_ext[1:]           # Last N samples (current MSB)
lsb = lsb_ext[1:]           # Last N samples (current LSB)
signal_kickback = msb + lsb + kickback_strength * msb_shifted

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

plt.sca(axes[0, 0])
result1 = analyze_spectrum(signal_noise, fs=Fs)
axes[0, 0].set_ylim([-120, 0])
axes[0, 0].set_title(f'Noise: RMS={noise_rms*1e6:.0f} uV')
print(f"[Noise   ] [ENOB] = {result1['enob']:.3f} b, [SNDR] = {result1['sndr_db']:.2f} dB, [SFDR] = {result1['sfdr_db']:.2f} dB, [SNR] = {result1['snr_db']:.2f} dB")

plt.sca(axes[0, 1])
result2 = analyze_spectrum(signal_jitter, fs=Fs)
axes[0, 1].set_ylim([-120, 0])
axes[0, 1].set_title(f'Jitter: {jitter_rms*1e15:.0f} fs')
print(f"[Jitter  ] [ENOB] = {result2['enob']:.3f} b, [SNDR] = {result2['sndr_db']:.2f} dB, [SFDR] = {result2['sfdr_db']:.2f} dB, [SNR] = {result2['snr_db']:.2f} dB")

plt.sca(axes[1, 0])
result3 = analyze_spectrum(signal_harmonic, fs=Fs)
axes[1, 0].set_ylim([-120, 0])
axes[1, 0].set_title(f'Harmonic Distortion: HD2 = {hd2_dB} dB, HD3 = {hd3_dB} dB')
print(f"[Harmonic] [ENOB] = {result3['enob']:.3f} b, [SNDR] = {result3['sndr_db']:.2f} dB, [SFDR] = {result3['sfdr_db']:.2f} dB, [SNR] = {result3['snr_db']:.2f} dB")

plt.sca(axes[1, 1])
result4 = analyze_spectrum(signal_kickback, fs=Fs)
axes[1, 1].set_ylim([-120, 0])
axes[1, 1].set_title(f'Kickback: strength = {kickback_strength}')
print(f"[Kickback] [ENOB] = {result4['enob']:.3f} b, [SNDR] = {result4['sndr_db']:.2f} dB, [SFDR] = {result4['sfdr_db']:.2f} dB, [SNR] = {result4['snr_db']:.2f} dB")

fig.suptitle(f'Spectrum Comparison: 4 Non-idealities (Fs = {Fs/1e6:.0f} MHz, Fin = {Fin/1e6:.1f} MHz)', fontsize=12, fontweight='bold')
plt.tight_layout()
fig_path = output_dir / 'exp_a01_analyze_spectrum_nonidealities.png'
plt.savefig(fig_path, dpi=150)
print(f"\n[Save fig] -> [{fig_path}]\n")
plt.close()
