"""
Time-interleave demo: compare FFT vs Farrow skew calibration.

Synthesizes a 4-channel TI-ADC capture with realistic offset / gain / skew
mismatch, measures the spectrum before and after calibration using each of the
two fractional-delay primitives, and plots the three spectra side by side.

Key observations to look for:
- Uncalibrated capture: spurs at ``k·fs/4`` (offset) and at ``fin ± k·fs/4``
  (gain+skew) dominate SFDR.
- FFT calibration: spurs drop into the noise-floor (numerical precision).
- Farrow calibration: spurs drop cleanly but not as low as FFT; residual is
  set by Lagrange truncation and by boundary transients of length ``n_taps//2``.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from adctoolbox import (
    analyze_spectrum,
    calibrate_foreground,
    extract_mismatch_sine,
    find_coherent_frequency,
)

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# ---------------- synthesize a mismatched TI capture ----------------

M = 4
Fs = 1e9
N_fft = 2**14
Fin, Fin_bin = find_coherent_frequency(Fs, 17e6, N_fft)
A = 0.5

rng = np.random.default_rng(42)
gain_true = 1.0 + 0.02 * rng.standard_normal(M)
offset_true = 0.005 * rng.standard_normal(M)
skew_true = 3e-12 * rng.standard_normal(M)
skew_true -= skew_true.mean()

T = 1.0 / Fs
x = np.empty(N_fft)
for n in range(N_fft):
    m = n % M
    t = n * T + skew_true[m]
    x[n] = gain_true[m] * A * np.cos(2 * np.pi * Fin * t) + offset_true[m]

print(f"[Setup] Fs={Fs/1e9:.1f} GHz, M={M}, Fin={Fin/1e6:.3f} MHz (bin {Fin_bin}/{N_fft})")
print(f"[Mismatch] gain spread     = {100*(gain_true.max()-gain_true.min()):.2f}% peak-to-peak")
print(f"[Mismatch] offset spread   = {1000*(offset_true.max()-offset_true.min()):.2f} mV peak-to-peak")
print(f"[Mismatch] skew spread     = {1e12*(skew_true.max()-skew_true.min()):.2f} ps peak-to-peak\n")

# ---------------- extract + calibrate ----------------

params = extract_mismatch_sine(x, M=M, fs=Fs, fin=Fin)
print(f"[Recovered gain]   {params['gain']}")
print(f"[Recovered offset] {params['offset']}")
print(f"[Recovered skew]   {params['skew'] * 1e12} ps\n")

x_fft = calibrate_foreground(x, M=M, params=params, fs=Fs, skew_method="fft")
x_far = calibrate_foreground(x, M=M, params=params, fs=Fs, skew_method="farrow", n_taps=9)

# ---------------- compare spectra ----------------

fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
for ax, sig, title in [
    (axes[0], x,     "Uncalibrated (TI spurs from offset + gain + skew)"),
    (axes[1], x_fft, "After calibrate_foreground(skew_method='fft')"),
    (axes[2], x_far, "After calibrate_foreground(skew_method='farrow', n_taps=9)"),
]:
    plt.sca(ax)
    res = analyze_spectrum(sig, fs=Fs)
    ax.set_title(
        f"{title}\n"
        f"SNDR={res['sndr_dbc']:.1f} dBc, SFDR={res['sfdr_dbc']:.1f} dBc, ENOB={res['enob']:.2f} b"
    )

plt.tight_layout()
fig_path = (output_dir / "exp_ti01_compare_skew_methods.png").resolve()
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"[Save fig] -> {fig_path}")
