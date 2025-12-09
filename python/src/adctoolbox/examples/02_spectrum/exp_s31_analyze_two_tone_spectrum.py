"""
Two-tone intermodulation distortion (IMD) analysis: measures IMD2 and IMD3 products.
IMD2 at |F1±F2|, IMD3 at |2F1-F2| and |2F2-F1|. Nonlinearity creates mixing products
that degrade SNDR. Demonstrates analyze_two_tone_spectrum for automatic IMD measurement.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import analyze_two_tone_spectrum, find_coherent_frequency, calculate_snr_from_amplitude, snr_to_nsd

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N_fft = 2**13
Fs = 1000e6
A1 = 0.5
A2 = 0.5
noise_rms = 100e-6

F1, bin_F1 = find_coherent_frequency(fs=Fs, fin_target=110e6, n_fft=N_fft)
F2, bin_F2 = find_coherent_frequency(fs=Fs, fin_target=100e6, n_fft=N_fft)

# For two-tone, combined RMS amplitude is sqrt((A1^2 + A2^2)/2)
A_combined_rms = np.sqrt((A1**2 + A2**2) / 2)
snr_ref = calculate_snr_from_amplitude(sig_amplitude=A_combined_rms, noise_amplitude=noise_rms)
nsd_ref = snr_to_nsd(snr_ref, fs=Fs, osr=1)
print(f"[Two-Tone] Fs=[{Fs/1e6:.1f} MHz], F1=[{F1/1e6:.2f} MHz] (Bin {bin_F1}), F2=[{F2/1e6:.2f} MHz] (Bin {bin_F2}), N=[{N_fft}]")
print(f"[Amplitude] A1=[{A1:.3f} Vpeak], A2=[{A2:.3f} Vpeak], Noise RMS=[{noise_rms*1e6:.2f} uVrms], Theoretical SNR=[{snr_ref:.2f} dB], Theoretical NSD=[{nsd_ref:.2f} dBFS/Hz]\n")

# Generate signal with nonlinearity and noise
t = np.arange(N_fft) / Fs
signal_base = A1 * np.sin(2*np.pi*F1*t) + A2 * np.sin(2*np.pi*F2*t)
signal = signal_base + 0.002 * signal_base**2 + 0.001 * signal_base**3 + np.random.randn(N_fft) * noise_rms

result = analyze_two_tone_spectrum(signal, fs=Fs)

print(f"[analyze_two_tone_spectrum] ENoB=[{result['enob']:5.2f} b], SNDR=[{result['sndr_db']:6.2f} dB], SFDR=[{result['sfdr_db']:6.2f} dB], SNR=[{result['snr_db']:6.2f} dB], IMD2=[{result['imd2_db']:6.2f} dB], IMD3=[{result['imd3_db']:6.2f} dB]")

fig_path = (output_dir / 'exp_s31_analyze_two_tone_spectrum.png').resolve()
print(f"\n[Save fig] -> [{fig_path}]")
plt.savefig(fig_path, dpi=150)
plt.close()
