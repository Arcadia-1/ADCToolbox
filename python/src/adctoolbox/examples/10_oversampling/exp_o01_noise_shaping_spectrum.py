"""Generate a noise-shaped ADC-like signal and analyze its in-band spectrum."""

from pathlib import Path

import matplotlib.pyplot as plt

from adctoolbox import analyze_spectrum, find_coherent_frequency
from adctoolbox.siggen import ADC_Signal_Generator


output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True, parents=True)

N = 2**13
Fs = 100e6
OSR = 32
Fin_target = Fs / (2 * OSR) / 5
Fin, J = find_coherent_frequency(Fs, Fin_target, N)

gen = ADC_Signal_Generator(N=N, Fs=Fs, Fin=Fin, A=0.4, DC=0.0)

signals = {
    "No shaping": gen.apply_quantization_noise(n_bits=10, quant_range=(-0.5, 0.5)),
    "1st-order NTF": gen.apply_noise_shaping(n_bits=10, quant_range=(-0.5, 0.5), order=1),
    "2nd-order NTF": gen.apply_noise_shaping(n_bits=10, quant_range=(-0.5, 0.5), order=2),
}

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
print(f"[Setup] Fs={Fs/1e6:.1f} MHz, Fin={Fin/1e6:.4f} MHz, bin={J}, OSR={OSR}")

for ax, (title, data) in zip(axes, signals.items()):
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
    print(
        f"{title:14s}: SNR={result['snr_dbc']:7.2f} dB, "
        f"SNDR={result['sndr_dbc']:7.2f} dB, ENOB={result['enob']:5.2f}"
    )

plt.tight_layout()
fig_path = output_dir / "exp_o01_noise_shaping_spectrum.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"[Save figure] -> {fig_path}")

