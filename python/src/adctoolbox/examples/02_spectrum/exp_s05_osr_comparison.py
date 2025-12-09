"""
Basic demo: Spectrum analysis with OSR sweep.

This script demonstrates the effect of oversampling ratio (OSR) on spectrum analysis
by sweeping through different OSR values and plotting the results.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import analyze_spectrum, find_coherent_frequency, amplitudes_to_snr, snr_to_nsd

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N_fft = 2**16
Fs = 100e6
Fin_target = 0.1e6
Fin, Fin_bin = find_coherent_frequency(fs=Fs, fin_target=Fin_target, n_fft=N_fft)
A = 0.5
noise_rms = 100e-6

snr_ref = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=noise_rms)
nsd_ref = snr_to_nsd(snr_ref, fs=Fs, osr=1)
print(f"[Sinewave] Fs=[{Fs/1e6:.2f} MHz], Fin=[{Fin/1e6:.2f} MHz], Bin/N=[{Fin_bin}/{N_fft}], A=[{A:.3f} Vpeak]")
print(f"[Nonideal] Noise RMS=[{noise_rms*1e6:.2f} uVrms], Theoretical SNR=[{snr_ref:.2f} dB], Theoretical NSD=[{nsd_ref:.2f} dBFS/Hz]\n")

t = np.arange(N_fft) / Fs
signal = A * np.sin(2*np.pi*Fin*t) + np.random.randn(N_fft) * noise_rms

# OSR sweep values
osr_values = [1, 2, 4, 10]

n_cols = len(osr_values)
fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 6, 5))
axes = axes.flatten()

for idx, osr in enumerate(osr_values):
    plt.sca(axes[idx])
    result = analyze_spectrum(signal, fs=Fs, osr=osr, show_plot=True)

    if idx == 0:
        snr_baseline = result['snr_db'] # Baseline SNR (OSR=1)

    snr_improvement = result['snr_db'] - snr_baseline    
    axes[idx].set_title(f'OSR = {osr} (SNR +{snr_improvement:.1f} dB)', fontsize=12, fontweight='bold')

    print(f"[OSR={osr:3d}] ENoB=[{result['enob']:5.2f} b], SNDR=[{result['sndr_db']:6.2f} dB], SFDR=[{result['sfdr_db']:6.2f} dB], SNR=[{result['snr_db']:6.2f} dB], NSD=[{result['nsd_dbfs_hz']:7.2f} dBFS/Hz], Delta SNR=[+{snr_improvement:.1f} dB]")

plt.tight_layout()
fig_path = (output_dir / 'exp_s05_osr_comparison.png').resolve()
print(f"\n[Save fig] -> [{fig_path}]\n")
plt.savefig(fig_path, dpi=150)
plt.close()