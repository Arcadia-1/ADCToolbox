"""Jitter spectrum comparison across Nyquist zones"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_bin, alias, spec_plot

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Parameters
N = 2**13
Fs = 800e6
A, DC, jitter = 0.49, 0.5, 100e-15

# 2x2 Spectrum Plots across 4 Nyquist zones
Fin_relative = [0.1, 0.9, 1.1, 1.9]
zone_labels = ['1st', '2nd', '3rd', '4th']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for i, (fin_rel, zone) in enumerate(zip(Fin_relative, zone_labels)):
    bin_idx = find_bin(Fs, fin_rel * Fs, N)
    fin_Hz = bin_idx * Fs / N
    t = np.arange(N) / Fs

    phase_jitter = np.random.randn(N) * 2 * np.pi * fin_Hz * jitter
    signal = A * np.sin(2 * np.pi * fin_Hz * t + phase_jitter) + DC

    plt.sca(axes.flatten()[i])
    spec_plot(signal, fs=Fs, harmonic=3, label=1)

    fin_alias_MHz = alias(fin_Hz, Fs) / 1e6
    axes.flatten()[i].set_title(f'Fs={Fs/1e6:.0f}MHz | Jitter={jitter*1e15:.0f}fs | Fin={fin_rel*Fs/1e6:.0f}MHz → Alias to {fin_alias_MHz:.1f}MHz ({zone} Nyquist Zone)', fontsize=10)
    axes.flatten()[i].set_xlim([0, Fs/2])
    axes.flatten()[i].set_ylim([-120, 0])

plt.tight_layout()
plt.savefig(output_dir / 'exp_a03_jitter_spectrum.png', dpi=150)
plt.close()

print(f"[Jitter Spectrum Comparison] [N={N}, Fs={Fs/1e6:.0f}MHz, Jitter={jitter*1e12:.1f}ps]")
for fin_rel in Fin_relative:
    print(f"  Fin={fin_rel*Fs/1e6:.0f}MHz → Alias to {alias(fin_rel * Fs, Fs) / 1e6:.1f}MHz")
print(f"[Save fig] -> [exp_a03_jitter_spectrum.png]")
