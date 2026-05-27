"""
4-bit SAR ADC demo: compare spectrum results across short and long FFT records.

The input tone is placed just below Nyquist for each FFT length. Even-length
records use ``N/2 - 1``; odd-length records use ``floor(N/2)``. For example,
``N=64`` uses ``Fin/Fs=31/64`` and ``N=65`` uses ``Fin/Fs=32/65``.

This example intentionally lets ``analyze_spectrum`` use its default
``side_bin`` behavior. For a rectangular window that default is one coherent
FFT bin; non-coherent captures should pass a larger ``side_bin`` explicitly.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from adctoolbox import analyze_spectrum
from adctoolbox.models import sar_convert, sar_ideal_weights, sar_reconstruct


output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

Fs = 100e6
N_BITS = 4
SAR_WEIGHTS = sar_ideal_weights(N_BITS)
FFT_LENGTHS = [4, 5, 8, 9, 16, 17, 32, 33, 64, 65, 128, 129, 256, 257]


def near_nyquist_bin(n_fft: int) -> int:
    """Return an integer input bin just below Nyquist."""
    if n_fft % 2 == 0:
        return max(1, n_fft // 2 - 1)
    return n_fft // 2


def format_metric(value: float, unit: str = "") -> str:
    """Readable print formatting for finite and NaN metrics."""
    if not np.isfinite(value):
        return "N/A"
    return f"{value:.2f}{unit}"


fig, axes = plt.subplots(7, 2, figsize=(14, 24.5), constrained_layout=True)
axes = axes.ravel()

results = []
for ax, n_fft in zip(axes, FFT_LENGTHS):
    fin_bin = near_nyquist_bin(n_fft)
    fin = fin_bin * Fs / n_fft
    t = np.arange(n_fft) / Fs

    vin = 0.5 + 0.49 * np.sin(2 * np.pi * fin * t)
    codes = sar_convert(vin, SAR_WEIGHTS, quant_range=(0.0, 1.0))
    aout = sar_reconstruct(codes, SAR_WEIGHTS, quant_range=(0.0, 1.0))

    metrics = analyze_spectrum(
        aout,
        fs=Fs,
        win_type="rectangular",
        nf_method=3,
        create_plot=True,
        show_title=False,
        show_label=True,
        plot_harmonics_up_to=3,
        ax=ax,
    )

    parity = "odd" if n_fft % 2 else "even"
    ax.set_title(
        f"N={n_fft} ({parity}), Fin/Fs={fin_bin}/{n_fft}",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_box_aspect(0.5)  # height / width = 1 / 2
    ax.tick_params(axis="both", labelsize=8)
    results.append((n_fft, fin_bin, metrics))

fig.suptitle(
    "4-bit SAR ADC Spectrum: Near-Nyquist FFT Length Cases",
    fontsize=15,
    fontweight="bold",
)

fig_path = (output_dir / "exp_s09_sar_fft_length_near_nyquist.png").resolve()
print(f"\n[Save fig] -> [{fig_path}]\n")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"{'N':>5} {'Fin/Fs':>12} {'Sig':>11} {'SNDR':>11} {'ENOB':>8} {'SNR':>11}")
print("-" * 65)
for n_fft, fin_bin, metrics in results:
    print(
        f"{n_fft:5d} "
        f"{fin_bin}/{n_fft:<8d} "
        f"{format_metric(metrics['sig_pwr_dbfs'], ' dBFS'):>11} "
        f"{format_metric(metrics['sndr_dbc'], ' dB'):>11} "
        f"{format_metric(metrics['enob']):>8} "
        f"{format_metric(metrics['snr_dbc'], ' dB'):>11}"
    )
