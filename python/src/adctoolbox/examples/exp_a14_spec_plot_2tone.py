"""Two-tone IMD analysis: F1 and F2 with nonlinearity creating IMD products"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import spec_plot_2tone

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Two-tone test parameters (per plan)
N = 8192
Fs = 1e6
F1 = 100e3
F2 = 120e3
A1, A2 = 0.45, 0.45
DC = 0.5

t = np.arange(N) / Fs

print(f"[Two-Tone IMD Analysis] [Fs = {Fs/1e3:.0f} kHz, F1 = {F1/1e3:.0f} kHz, F2 = {F2/1e3:.0f} kHz, N = {N}]")

# Generate two-tone signal with nonlinearity
np.random.seed(42)
noise_rms = 1e-4

# Two tones
signal_base = A1 * np.sin(2*np.pi*F1*t) + A2 * np.sin(2*np.pi*F2*t)

# Add 2nd order nonlinearity (creates IMD2 products)
k2 = 0.02
signal_with_imd = signal_base + k2 * signal_base**2

# Add 3rd order nonlinearity (creates IMD3 products)
k3 = 0.01
signal_with_imd += k3 * signal_base**3

# Add DC and noise
signal = signal_with_imd + DC + np.random.randn(N) * noise_rms

# Analyze two-tone spectrum
enob, sndr, sfdr, snr, thd, pwr1, pwr2, nf, imd2, imd3 = spec_plot_2tone(
    signal, fs=Fs, harmonic=7, is_plot=False
)

print(f"\n[Two-Tone Metrics]")
print(f"  ENOB:  {enob:.2f} bits")
print(f"  SNDR:  {sndr:.2f} dB")
print(f"  SFDR:  {sfdr:.2f} dB")
print(f"  SNR:   {snr:.2f} dB")
print(f"  THD:   {thd:.2f} dB")
print(f"  IMD2:  {imd2:.2f} dB")
print(f"  IMD3:  {imd3:.2f} dB")
print(f"  Tone1: {pwr1:.2f} dBFS @ {F1/1e3:.0f} kHz")
print(f"  Tone2: {pwr2:.2f} dBFS @ {F2/1e3:.0f} kHz")
print(f"  Noise: {nf:.2f} dB")

# Create custom plot showing key IMD products
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# Compute spectrum
window = np.hanning(N)
spec = np.fft.fft(signal * window)
spec_mag = np.abs(spec[:N//2])
spec_dB = 20 * np.log10(spec_mag / (np.max(spec_mag) + 1e-10))
freq = np.arange(N//2) * Fs / N

# Plot spectrum
ax.plot(freq/1e3, spec_dB, 'b-', linewidth=0.8, alpha=0.7)
ax.set_xlabel('Frequency (kHz)', fontsize=12)
ax.set_ylabel('Magnitude (dBFS)', fontsize=12)
ax.set_title(f'Two-Tone Spectrum: F1={F1/1e3:.0f}kHz, F2={F2/1e3:.0f}kHz\nIMD2={imd2:.1f}dB, IMD3={imd3:.1f}dB, SFDR={sfdr:.1f}dB',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim([-120, 10])
ax.set_xlim([0, Fs/2/1e3])

# Mark key frequencies
# F1 and F2
ax.axvline(F1/1e3, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='F1, F2')
ax.axvline(F2/1e3, color='r', linestyle='--', linewidth=1.5, alpha=0.7)

# IMD2 products: F2-F1, F2+F1
imd2_low = (F2 - F1) / 1e3
imd2_high = (F2 + F1) / 1e3
ax.axvline(imd2_low, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='IMD2')
ax.axvline(imd2_high, color='orange', linestyle=':', linewidth=2, alpha=0.7)

# IMD3 products: 2F1-F2, 2F2-F1
imd3_low = (2*F1 - F2) / 1e3
imd3_high = (2*F2 - F1) / 1e3
ax.axvline(imd3_low, color='green', linestyle='-.', linewidth=2, alpha=0.7, label='IMD3')
ax.axvline(imd3_high, color='green', linestyle='-.', linewidth=2, alpha=0.7)

ax.legend(fontsize=10, loc='upper right')

plt.tight_layout()
fig_path = output_dir / f'exp_a14_spec_plot_2tone_f1_{int(F1/1e3)}k_f2_{int(F2/1e3)}k.png'
plt.savefig(fig_path, dpi=150)
plt.close()

print(f"\n[IMD Product Frequencies]")
print(f"  F1: {F1/1e3:.0f} kHz")
print(f"  F2: {F2/1e3:.0f} kHz")
print(f"  IMD2: {imd2_low:.0f} kHz (F2-F1), {imd2_high:.0f} kHz (F2+F1)")
print(f"  IMD3: {imd3_low:.0f} kHz (2F1-F2), {imd3_high:.0f} kHz (2F2-F1)")

print(f"\n[Save fig] -> [{fig_path}]")
