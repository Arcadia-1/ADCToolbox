import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import analyze_spectrum

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N_fft = 2**13
Fs = 100e6
A = 0.5
noise_rms = 50e-6

Fin = 10e6

print(f"[Window Comparison - Spectral Leakage]")
print(f"[N_fft] = {N_fft}, [Fs] = {Fs/1e6:.2f} MHz")
print(f"[Fin] = {Fin/1e6:.3f} MHz (non-coherent)\n")

windows = ['boxcar', 'hann', 'hamming', 'blackman', 'blackmanharris', 'flattop', 'kaiser', 'chebwin']
window_descriptions = {
    'boxcar': 'Rectangular (no window)', 'hann': 'Hann (raised cosine)',
    'hamming': 'Hamming', 'blackman': 'Blackman',
    'blackmanharris': 'Blackman-Harris', 'flattop': 'Flat-top',
    'kaiser': 'Kaiser (beta=38)', 'chebwin': 'Chebyshev (100 dB)'
}
window_side_bins = {
    'boxcar': 1, 'hann': 2, 'hamming': 2, 'blackman': 3,
    'blackmanharris': 4, 'flattop': 4, 'kaiser': 8, 'chebwin': 4
}

t = np.arange(N_fft) / Fs
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for idx, win_type in enumerate(windows):
    signal = A * np.sin(2*np.pi*Fin*t) + np.random.randn(N_fft) * noise_rms
    result = analyze_spectrum(signal, fs=Fs, win_type=win_type,
                             side_bin=window_side_bins[win_type],
                             plot_harmonics_up_to=7, ax=axes[idx])

    axes[idx].set_title(f'{window_descriptions[win_type]} Window', fontsize=12, fontweight='bold')
    axes[idx].set_ylim([-140, 0])
    print(f"[{win_type:8s}] [ENOB = {result['enob']:5.2f} b] [SNDR = {result['sndr_db']:5.2f} dB] [SFDR = {result['sfdr_db']:6.2f} dB]")

fig.suptitle(f'Spectral Leakage: Window Comparison (Fin={Fin/1e6:.1f} MHz, N={N_fft})',
             fontsize=14, fontweight='bold')
plt.tight_layout()
fig_path = output_dir / 'exp_s06_window_leakage.png'
print(f"\n[Save fig] -> [{fig_path}]")
plt.savefig(fig_path, dpi=150)
plt.close()

print("\n[Observation] With spectral leakage, window choice is CRITICAL")
print("[Boxcar] ENOB ~2 bits - severe leakage, unusable")
print("[Hann/Hamming] ENOB ~6 bits - moderate improvement")
print("[Kaiser/Blackman-Harris] ENOB ~12 bits - excellent leakage suppression")
