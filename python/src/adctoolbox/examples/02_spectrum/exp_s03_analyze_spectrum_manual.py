"""Manual spectrum analysis example using modular calculation and plotting."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, calculate_snr_from_amplitude, snr_to_nsd
from adctoolbox.spectrum import compute_spectrum, plot_spectrum

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N_fft = 2**13
Fs = 100e6
Fin_target = 12e6
Fin, Fin_bin = find_coherent_frequency(fs=Fs, fin_target=Fin_target, n_fft=N_fft)
A = 0.5
noise_rms = 200e-6

snr_ref = calculate_snr_from_amplitude(sig_amplitude=A, noise_amplitude=noise_rms)
nsd_ref = snr_to_nsd(snr_ref, fs=Fs, osr=1)
print(f"[Sinewave] Fs=[{Fs/1e6:.2f} MHz], Fin=[{Fin/1e6:.2f} MHz], Bin/N=[{Fin_bin}/{N_fft}], A=[{A:.3f} Vpeak]")
print(f"[Nonideal] Noise RMS=[{noise_rms*1e6:.2f} uVrms], Theoretical SNR=[{snr_ref:.2f} dB], Theoretical NSD=[{nsd_ref:.2f} dBFS/Hz]\n")


t = np.arange(N_fft) / Fs
signal = A * np.sin(2*np.pi*Fin*t) + np.random.randn(N_fft) * noise_rms

# Step 1: Calculate spectrum metrics (pure computation)
results = compute_spectrum(signal, fs=Fs)
metrics = results['metrics']

# Step 2: Display results
print(f"[compute_spectrum] ENoB=[{metrics['enob']:5.2f} b], SNDR=[{metrics['sndr_db']:6.2f} dB], SFDR=[{metrics['sfdr_db']:6.2f} dB], SNR=[{metrics['snr_db']:6.2f} dB], NSD=[{metrics['nsd_dbfs_hz']:7.2f} dBFS/Hz]")

# Step 3: Plot the spectrum (pure visualization)
fig, ax = plt.subplots(figsize=(8, 6))
plot_spectrum(results, ax=ax)

# Step 4: Save figure
fig_path = (output_dir / 'exp_s03_analyze_spectrum_manual.png').resolve()
print(f"\n[Save fig] -> [{fig_path}]\n")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()