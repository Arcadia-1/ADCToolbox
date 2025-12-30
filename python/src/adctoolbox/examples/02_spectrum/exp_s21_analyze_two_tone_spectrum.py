"""
Two-tone intermodulation distortion (IMD) analysis: measures IMD2 and IMD3 products.
IMD2 at |F1±F2|, IMD3 at |2F1-F2| and |2F2-F1|. Nonlinearity creates mixing products
that degrade SNDR. Demonstrates analyze_two_tone_spectrum for automatic IMD measurement.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import analyze_two_tone_spectrum, find_coherent_frequency, amplitudes_to_snr, snr_to_nsd

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N_fft = 2**13
Fs = 1000e6
A1 = 0.5
A2 = 0.5
noise_rms = 100e-6

F1, bin_F1 = find_coherent_frequency(fs=Fs, fin_target=410e6, n_fft=N_fft)
F2, bin_F2 = find_coherent_frequency(fs=Fs, fin_target=400e6, n_fft=N_fft)

# Calculate combined signal amplitude for two tones
# For two tones: total_power = A1²/2 + A2²/2, total_rms = √(A1²/2 + A2²/2)
# amplitudes_to_snr expects peak amplitude and converts to RMS by dividing by √2
# So we need equivalent peak: A_eq = total_rms * √2 = √(A1² + A2²)
sig_amplitude = np.sqrt(A1**2 + A2**2)
snr_ref = amplitudes_to_snr(sig_amplitude=sig_amplitude, noise_amplitude=noise_rms)
nsd_ref = snr_to_nsd(snr_ref, fs=Fs)

print(f"[Sinewave] Fs=[{Fs/1e6:.1f} MHz], F1=[{F1/1e6:.2f} MHz] (Bin/N={bin_F1}/{N_fft}), F2=[{F2/1e6:.2f} MHz] (Bin/N={bin_F2}/{N_fft})")
print(f"[Sinewave] A1=[{A1:.3f} Vpeak], A2=[{A2:.3f} Vpeak]")
print(f"[Nonideal] Noise RMS=[{noise_rms*1e6:.2f} uVrms], Theoretical SNR=[{snr_ref:.2f} dB], Theoretical NSD=[{nsd_ref:.2f} dBFS/Hz]\n")

t = np.arange(N_fft) / Fs

signal = A1 * np.sin(2*np.pi*F1*t) + A2 * np.sin(2*np.pi*F2*t) + np.random.randn(N_fft) * noise_rms

fig, ax = plt.subplots(figsize=(8, 6))
result = analyze_two_tone_spectrum(signal, fs=Fs, ax=ax)

print(f"[two_tone] ENoB=[{result['enob']:5.2f} b], SNDR=[{result['sndr_dbc']:6.2f} dB], SFDR=[{result['sfdr_dbc']:6.2f} dB], SNR=[{result['snr_dbc']:6.2f} dB], IMD2=[{result['imd2_dbc']:6.2f} dB], IMD3=[{result['imd3_dbc']:6.2f} dB]")

fig_path = (output_dir / 'exp_s21_analyze_two_tone_spectrum.png').resolve()
print(f"\n[Save fig] -> [{fig_path}]")
plt.savefig(fig_path, dpi=150)
plt.close()
