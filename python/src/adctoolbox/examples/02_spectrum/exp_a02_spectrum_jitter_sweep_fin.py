"""Spectrum comparison: jitter_rms across Nyquist zones"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, fold_frequency_to_nyquist, analyze_spectrum, amplitudes_to_snr, snr_to_nsd

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N = 2**13
Fs = 10e9
A = 0.49
jitter_rms = 50e-15
base_noise = 1e-6

# Four frequencies across 4 Nyquist zones - all alias to 1GHz
Fin_list = [1e9, 9e9, 11e9, 19e9]
zone_labels = ['1st', '2nd', '3rd', '4th']

print(f"[Sinewave] Fs=[{Fs/1e9:.1f} GHz], N=[{N}], A=[{A:.3f} Vpeak]")
print(f"[Nonideal] Jitter=[{jitter_rms*1e15:.1f} fs], Base Noise=[{base_noise*1e6:.2f} uVrms]\n")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i, (fin, zone) in enumerate(zip(Fin_list, zone_labels)):
    fin_coherent, bin = find_coherent_frequency(fs=Fs, fin_target=fin, n_fft=N)
    fin_alias = fold_frequency_to_nyquist(fin=fin_coherent, fs=Fs)

    # Calculate theoretical SNR and NSD
    # Jitter noise power: (2*pi*fin*jitter_rms)^2 / 2
    jitter_noise_rms = 2 * np.pi * fin_coherent * jitter_rms * A / np.sqrt(2)
    total_noise_rms = np.sqrt(base_noise**2 + jitter_noise_rms**2)
    snr_ref = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=total_noise_rms)
    nsd_ref = snr_to_nsd(snr_ref, fs=Fs, osr=1)

    t = np.arange(N) / Fs
    phase_jitter = np.random.randn(N) * 2 * np.pi * fin_coherent * jitter_rms
    signal = A * np.sin(2*np.pi*fin_coherent*t + phase_jitter) + np.random.randn(N) * base_noise

    row, col = i // 2, i % 2
    plt.sca(axes[row, col])
    result = analyze_spectrum(signal, fs=Fs)
    axes[row, col].set_ylim([-120, 0])

    print(f"[{zone:4s} Zone] Fin=[{fin_coherent/1e9:.2f} GHz] (Bin/N=[{bin}/{N}]) → Alias=[{fin_alias/1e9:.3f} GHz]")
    print(f"  [Theoretical] SNR=[{snr_ref:.2f} dB], NSD=[{nsd_ref:.2f} dBFS/Hz]")
    print(f"  [Measured] ENoB=[{result['enob']:5.2f} b], SNDR=[{result['sndr_db']:6.2f} dB], SFDR=[{result['sfdr_db']:6.2f} dB], SNR=[{result['snr_db']:6.2f} dB], NSD=[{result['nsd_dbfs_hz']:7.2f} dBFS/Hz]")

    fin_GHz = fin_coherent / 1e9
    fin_alias_GHz = fin_alias / 1e9
    axes[row, col].set_title(f'{zone} Nyquist Zone: Fin = {fin_GHz:.2f} GHz → {fin_alias_GHz:.3f} GHz')

fig.suptitle(f'Jitter Across Nyquist Zones (Jitter = {jitter_rms*1e15:.0f}fs, Fs = {Fs/1e9:.1f} GHz)', fontsize=12, fontweight='bold')
plt.tight_layout()
fig_path = output_dir / 'exp_a02_spectrum_jitter_sweep_fin.png'
print(f"\n[Save fig] -> [{fig_path}]")
plt.savefig(fig_path, dpi=150)
plt.close()