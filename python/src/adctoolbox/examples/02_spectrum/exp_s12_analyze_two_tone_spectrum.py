"""Two-tone IMD analysis demonstrating IMD2/IMD3 products.

Demonstrates coherent two-tone signal generation with nonlinearity.
Shows how 2nd and 3rd order distortion creates intermodulation products.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import analyze_two_tone_spectrum, calculate_coherent_freq

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N_fft = 2**13
Fs = 1000e6
A1 = 0.5
A2 = 0.5
noise_rms = 100e-6

# Calculate coherent frequencies (odd bins, coprime with N)
F1, bin_F1 = calculate_coherent_freq(fs=Fs, fin_target=110e6, n_fft=N_fft)
F2, bin_F2 = calculate_coherent_freq(fs=Fs, fin_target=100e6, n_fft=N_fft)

print(f"[Two-Tone IMD Analysis]")
print(f"  Fs={Fs/1e3:.0f} kHz, N={N_fft}")
print(f"  F1={F1/1e3:.3f} kHz @ bin {bin_F1}")
print(f"  F2={F2/1e3:.3f} kHz @ bin {bin_F2}")

# Generate signal with nonlinearity and noise
t = np.arange(N_fft) / Fs
signal_base = A1 * np.sin(2*np.pi*F1*t) + A2 * np.sin(2*np.pi*F2*t)
signal = signal_base + 0.002 * signal_base**2 + 0.001 * signal_base**3 + np.random.randn(N_fft) * noise_rms

# Analyze two-tone spectrum
result = analyze_two_tone_spectrum(signal, fs=Fs)

print(f"\n[Metrics]")
print(f"  ENOB={result['enob']:.2f} bits, SNDR={result['sndr_db']:.2f} dB, SFDR={result['sfdr_db']:.2f} dB")
print(f"  SNR={result['snr_db']:.2f} dB, THD={result['thd_db']:.2f} dB")
print(f"  IMD2={result['imd2_db']:.2f} dBc, IMD3={result['imd3_db']:.2f} dBc")
print(f"  Tone1={result['signal_power_1_dbfs']:.2f} dBFS, Tone2={result['signal_power_2_dbfs']:.2f} dBFS")

print(f"\n[IMD Products]")
print(f"  IMD2: {abs(F2-F1)/1e3:.2f} kHz (F2-F1), {(F2+F1)/1e3:.2f} kHz (F2+F1)")
print(f"  IMD3: {abs(2*F1-F2)/1e3:.2f} kHz (2F1-F2), {abs(2*F2-F1)/1e3:.2f} kHz (2F2-F1)")

fig_path = output_dir / 'exp_s12_analyze_two_tone_spectrum.png'
print(f"\n[Save fig] -> [{fig_path}]")
plt.savefig(fig_path, dpi=150)
plt.close()
