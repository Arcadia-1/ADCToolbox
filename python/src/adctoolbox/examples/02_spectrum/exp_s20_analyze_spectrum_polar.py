"""
Polar phase spectrum analysis: visualize FFT bins as magnitude-phase vectors in polar coordinates.
Demonstrates ideal case with only thermal noise - shows random phase distribution for noise floor
and clear signal phase at fundamental frequency.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, amplitudes_to_snr, snr_to_nsd, analyze_spectrum_polar

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N = 2**13
Fs = 800e6
Fin_target = 80e6
Fin, Fin_bin = find_coherent_frequency(Fs, Fin_target, N)
t = np.arange(N) / Fs
A, DC = 0.49, 0.5
print(f"[Sinewave] Fs=[{Fs/1e6:.2f} MHz], Fin=[{Fin/1e6:.2f} MHz], Bin/N=[{Fin_bin}/{N}], A=[{A:.3f} Vpeak]")

# Three noise levels to show effect on phase clarity
noise_levels = [50e-6, 500e-6, 2e-3]
noise_labels = ['50 uVrms (Low noise)', '500 uVrms (Medium noise)', '2 mVrms (High noise)']

fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': 'polar'})

for i, (noise_rms, label) in enumerate(zip(noise_levels, noise_labels)):
    # Generate ideal sinewave with thermal noise only
    sig_ideal = A * np.sin(2*np.pi*Fin*t)
    signal = sig_ideal + DC + np.random.randn(N) * noise_rms

    # Calculate theoretical values for this noise level
    snr_theory = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=noise_rms)
    nsd_theory = snr_to_nsd(snr_theory, fs=Fs, osr=1)

    plt.sca(axes[i])
    result = analyze_spectrum_polar(signal, fs=Fs, fixed_radial_range=120)
    axes[i].set_title(f'{label}', pad=20, fontsize=12, fontweight='bold')

    print(f"\n[{label}]")    
    print(f"[Theory] SNR=[{snr_theory:6.2f} dB], NSD=[{nsd_theory:7.2f} dBFS/Hz]")
    print(f"[analyze_spectrum_polar] ENoB=[{result['enob']:5.2f} b], SNDR=[{result['sndr_db']:6.2f} dB], SFDR=[{result['sfdr_db']:6.2f} dB], SNR=[{result['snr_db']:6.2f} dB], NSD=[{result['nsd_dbfs_hz']:7.2f} dBFS/Hz]")

plt.tight_layout()
fig_path = output_dir / 'exp_s20_analyze_spectrum_polar.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"\n[Save fig] -> [{fig_path}]")
plt.close()
