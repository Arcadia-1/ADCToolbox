"""Spectrum comparison: jitter_rms across Nyquist zones"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import calc_coherent_freq, calc_aliased_freq, analyze_spectrum

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N = 2**13
Fs = 10e9
A = 0.49
jitter_rms = 50e-15
base_noise = 1e-6

# Four frequencies across 4 Nyquist zones - all alias to 1GHz
Fin_list = [1e9,9e9,11e9,19e9]
zone_labels = ['1st', '2nd', '3rd', '4th']

print(f"[Jitter Across Nyquist Zones] [Fs = {Fs/1e9:.1f} GHz, Jitter = {jitter_rms*1e15:.1f} fs, N = {N}]")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i, (fin, zone) in enumerate(zip(Fin_list, zone_labels)):
    fin_coherent, bin = calc_coherent_freq(fs=Fs, fin_target=fin, n_fft=N)
    t = np.arange(N) / Fs

    phase_jitter = np.random.randn(N) * 2 * np.pi * fin_coherent * jitter_rms
    signal = A * np.sin(2*np.pi*fin_coherent*t + phase_jitter) + np.random.randn(N) * base_noise

    row, col = i // 2, i % 2
    plt.sca(axes[row, col])
    result = analyze_spectrum(signal, fs=Fs)
    axes[row, col].set_ylim([-120, 0])

    fin_GHz = fin_coherent / 1e9
    fin_alias_GHz = calculate_aliased_freq(fin=fin_coherent, fs=Fs) / 1e9
    axes[row, col].set_title(f'{zone} Nyquist Zone: Fin = {fin_GHz:.2f} GHz → {fin_alias_GHz:.3f} GHz')

    # Print info for this zone
    print(f"  [{zone} Zone] Fin = {fin_GHz:4.1f} GHz -> Alias to {fin_alias_GHz:3.1f} GHz")

fig.suptitle(f'Jitter Across Nyquist Zones (Jitter = {jitter_rms*1e15:.0f}fs, Fs = {Fs/1e9:.1f} GHz)', fontsize=12, fontweight='bold')
plt.tight_layout()
fig_path = output_dir / 'exp_a02_analyze_spectrum_jitter.png'
plt.savefig(fig_path, dpi=150)
plt.close()

print(f"[Save fig] -> [{fig_path}]")