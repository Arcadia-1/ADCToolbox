"""
Demonstrate spectral leakage effects with 8 window functions on non-coherent sampling.
Rectangular: ~2b ENOB (severe leakage with wide skirts). Hann/Hamming: ~6b ENOB (moderate suppression).
Blackman: ~9.5b ENOB (good). Blackman-Harris/Flat-top/Kaiser/Chebyshev: ~12b ENOB (excellent).
Rule: For non-coherent sampling, use Kaiser/Blackman-Harris for best leakage suppression.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import analyze_spectrum, amplitudes_to_snr, snr_to_nsd

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N_fft = 2**13
Fs = 100e6
A = 0.5
noise_rms = 50e-6
Fin = 10e6

snr_ref = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=noise_rms)
nsd_ref = snr_to_nsd(snr_ref, fs=Fs, osr=1)
print(f"[Sinewave] Fs=[{Fs/1e6:.2f} MHz], Fin=[{Fin/1e6:.2f} MHz] (non-coherent), N=[{N_fft}], A=[{A:.3f} Vpeak]")
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

fig.suptitle(f'Spectral Leakage: Window Comparison (Fin={Fin/1e6:.1f} MHz, N={N_fft})',
             fontsize=14, fontweight='bold')
plt.tight_layout()
fig_path = output_dir / 'exp_s07_window_leakage.png'
print(f"\n[Save fig] -> [{fig_path}]")
plt.savefig(fig_path, dpi=150)
plt.close()
