"""Spectrum analysis showing harmonic aliasing collision vs non-collision cases.

This example demonstrates:
- Fin = Fs/4: HD2, HD6, HD10 collide at Nyquist (50 MHz)
- Fin = Fs/17.6: No harmonic collisions (all harmonics spread across spectrum)
- Compares n_thd=3 vs n_thd=11 for both frequencies
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, amplitudes_to_snr, snr_to_nsd, analyze_spectrum

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N_fft = 2**19
Fs = 100e6
A = 0.5
noise_rms = 50e-6
hd2_dB = -100
hd3_dB = -80

snr_ref = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=noise_rms)
nsd_ref = snr_to_nsd(snr_ref, fs=Fs, osr=1)
print(f"[Parameters] Fs=[{Fs/1e6:.2f} MHz], A=[{A:.3f} Vpeak]")
print(f"[Nonideal] HD2=[{hd2_dB} dB], HD3=[{hd3_dB} dB], Noise RMS=[{noise_rms*1e6:.2f} uVrms], Theoretical SNR=[{snr_ref:.2f} dB], Theoretical NSD=[{nsd_ref:.2f} dBFS/Hz]\n")

# Compute nonlinearity coefficients to achieve target HD levels
# HD2: k2 * A^2 / 2 = hd2_amp * A  →  k2 = hd2_amp / (A/2)
# HD3: k3 * A^3 / 4 = hd3_amp * A  →  k3 = hd3_amp / (A^2/4)
hd2_amp = 10**(hd2_dB/20)
hd3_amp = 10**(hd3_dB/20)
k2 = hd2_amp / (A / 2)
k3 = hd3_amp / (A**2 / 4)

# Two test frequencies: collision vs non-collision
freq_configs = [
    {'target': Fs / 3, 'name': 'Fs/3', 'note': 'Collide at Nyquist!'},
    {'target': Fs / 17.6, 'name': 'Fs/17.6 (No Collision)', 'note': 'Harmonics spread'}
]

harmonic_counts = [3, 11]

# Create 2x2 grid: 2 frequencies × 2 harmonic counts
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for freq_idx, freq_config in enumerate(freq_configs):
    Fin, Fin_bin = find_coherent_frequency(fs=Fs, fin_target=freq_config['target'], n_fft=N_fft)
    print(f"[Frequency {freq_idx+1}] {freq_config['name']}: Fin=[{Fin/1e6:.2f} MHz], Bin/N=[{Fin_bin}/{N_fft}], {freq_config['note']}")

    # Generate signal with harmonics
    t = np.arange(N_fft) / Fs
    sig_ideal = A * np.sin(2 * np.pi * Fin * t)
    signal = sig_ideal + k2 * sig_ideal**2 + k3 * sig_ideal**3 + np.random.randn(N_fft) * noise_rms

    for harm_idx, n_harm in enumerate(harmonic_counts):
        ax = axes[freq_idx, harm_idx]
        plt.sca(ax)
        metrics = analyze_spectrum(signal, fs=Fs, n_thd=n_harm, show_title=False, plot_harmonics_up_to=n_harm)

        ax.set_title(f'{freq_config["name"]}, Harmonics up to {n_harm}', fontsize=12, fontweight='bold')
        print(f"  [n_thd={n_harm:2d}] ENoB=[{metrics['enob']:5.2f} b], SNDR=[{metrics['sndr_db']:6.2f} dB], THD=[{metrics['thd_db']:7.2f} dB]")

    print()

plt.tight_layout()

# Save figure
fig_path = (output_dir / 'exp_s05_annotating_spur.png').resolve()
print(f"\n[Save fig] -> [{fig_path}]")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()