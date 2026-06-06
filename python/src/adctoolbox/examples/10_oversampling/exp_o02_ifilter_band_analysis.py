"""Use MATLAB-compatible ifilter to isolate an oversampled ADC signal band."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from adctoolbox import analyze_spectrum, find_coherent_frequency, ifilter
from adctoolbox.siggen import ADC_Signal_Generator


output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True, parents=True)

N = 2**13
Fs = 100e6
OSR = 32
Fin_target = Fs / (2 * OSR) / 5
Fin, _ = find_coherent_frequency(Fs, Fin_target, N)

gen = ADC_Signal_Generator(N=N, Fs=Fs, Fin=Fin, A=0.4, DC=0.0)
sig = gen.apply_noise_shaping(n_bits=10, quant_range=(-0.5, 0.5), order=2)

t = np.arange(N) / Fs
sig = sig + 0.01 * np.sin(2 * np.pi * 0.38 * Fs * t)

sig_ib = ifilter(sig, [[0, 0.5 / OSR]]).ravel()
sig_oob = ifilter(sig, [[0.30, 0.45]]).ravel()

print(
    "RMS original / in-band / high-band: "
    f"{np.std(sig):.4e} / {np.std(sig_ib):.4e} / {np.std(sig_oob):.4e}"
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
for ax, title, data in [
    (axes[0], "Original with out-of-band tone", sig),
    (axes[1], "After ifilter in-band extraction", sig_ib),
]:
    result = analyze_spectrum(
        data,
        fs=Fs,
        osr=OSR,
        max_scale_range=[-0.5, 0.5],
        create_plot=True,
        ax=ax,
        show_title=False,
    )
    ax.set_title(title)
    ax.set_ylim([-170, 5])
    print(f"{title}: SNDR={result['sndr_dbc']:.2f} dB, SFDR={result['sfdr_dbc']:.2f} dB")

plt.tight_layout()
fig_path = output_dir / "exp_o02_ifilter_band_analysis.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"[Save figure] -> {fig_path}")

