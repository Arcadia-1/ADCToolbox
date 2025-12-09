"""
Demonstrate window functions with coherent sampling: most windows achieve ~12.5b ENOB, SFDR >103 dB.
No spectral leakage - signal sits perfectly in one bin. Rectangular/Hann/Hamming/Blackman/Blackman-Harris
all perform excellently (~12.5b ENOB). Only Chebyshev shows slight degradation (12.39b ENOB, SFDR 96 dB).
Rule: For coherent sampling, Rectangular/Hann/Hamming/Blackman/Blackman-Harris all work equally well.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import calculate_coherent_freq, analyze_spectrum, calculate_snr_from_amplitude, snr_to_nsd

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N_fft = 2**13
Fs = 100e6
A = 0.5
noise_rms = 50e-6

Fin_target = 10e6
Fin, Fin_bin = calculate_coherent_freq(Fs, Fin_target, N_fft)

snr_ref = calculate_snr_from_amplitude(sig_amplitude=A, noise_amplitude=noise_rms)
nsd_ref = snr_to_nsd(snr_ref, fs=Fs, osr=1)
print(f"[Sinewave] Fs=[{Fs/1e6:.2f} MHz], Fin=[{Fin/1e6:.6f} MHz] (coherent, Bin {Fin_bin}), N=[{N_fft}], A=[{A:.3f} Vpeak]")
print(f"[Nonideal] Noise RMS=[{noise_rms*1e6:.2f} uVrms], Theoretical SNR=[{snr_ref:.2f} dB], Theoretical NSD=[{nsd_ref:.2f} dBFS/Hz]\n")

WINDOW_CONFIGS = {
    'rectangular': {'description': 'Rectangular (no window)', 'side_bins': 1},
    'hann': {'description': 'Hann (raised cosine)', 'side_bins': 2},
    'hamming': {'description': 'Hamming', 'side_bins': 2},
    'blackman': {'description': 'Blackman', 'side_bins': 3},
    'blackmanharris': {'description': 'Blackman-Harris', 'side_bins': 4},
    'flattop': {'description': 'Flat-top', 'side_bins': 4},
    'kaiser': {'description': 'Kaiser (beta=38)', 'side_bins': 8},
    'chebwin': {'description': 'Chebyshev (100 dB)', 'side_bins': 4}
}

t = np.arange(N_fft) / Fs
signal = A * np.sin(2*np.pi*Fin*t) + np.random.randn(N_fft) * noise_rms

n_cols = 4
n_rows = 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
axes = axes.flatten()

for idx, win_type in enumerate(WINDOW_CONFIGS.keys()):
    plt.sca(axes[idx])
    result = analyze_spectrum(signal, fs=Fs, win_type=win_type, side_bin=WINDOW_CONFIGS[win_type]['side_bins'])
    axes[idx].set_ylim([-140, 0])
    axes[idx].set_title(f'{WINDOW_CONFIGS[win_type]["description"]} Window', fontsize=12, fontweight='bold')

    print(f"[{win_type:14s}] ENoB=[{result['enob']:5.2f} b], SNDR=[{result['sndr_db']:6.2f} dB], SFDR=[{result['sfdr_db']:6.2f} dB], SNR=[{result['snr_db']:6.2f} dB], NSD=[{result['nsd_dbfs_hz']:7.2f} dBFS/Hz]")

fig.suptitle(f'Coherent Sampling: Window Comparison (Fin={Fin/1e6:.6f} MHz, Bin {Fin_bin}, N={N_fft})',
             fontsize=14, fontweight='bold')
plt.tight_layout()
fig_path = output_dir / 'exp_s08_window_coherent.png'
print(f"\n[Save fig] -> [{fig_path}]")
plt.savefig(fig_path, dpi=150)
plt.close()
