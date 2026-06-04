"""
FFT-length Monte Carlo sweep: SFDR and SNDR with a fixed HD3 spur.

This example models a 10-bit ADC whose additive noise gives about 9 effective
bits, then adds a deterministic third harmonic at -80 dBc. For each FFT length
from 2^4 to 2^16, it runs 100 Monte Carlo captures and plots the SFDR/SNDR
spread.

The main observation is that SFDR improves with FFT length while the largest
noise bin dominates, then flattens near 80 dBc once the fixed HD3 spur becomes
the limiting tone. SNDR stays close to the target set by the wideband noise.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from adctoolbox import analyze_spectrum


output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N_BITS = 10
TARGET_ENOB = 9.0
N_MC = 100
Fs = 800e6
A = 0.49
DC = 0.5
HD3_DBC = -80.0
FIN_RATIO = 0.123
RNG_SEED = 20260604


def coherent_odd_bin(n_fft: int, target_ratio: float) -> int:
    """Pick an odd coherent input bin near target_ratio * Fs."""
    k = int(round(target_ratio * n_fft))
    k = min(max(k, 1), n_fft // 2 - 1)
    if k % 2 == 0:
        k += 1 if k + 1 < n_fft // 2 else -1
    return max(k, 1)


def quantize_ideal(vin: np.ndarray, n_bits: int) -> np.ndarray:
    """Ideal uniform ADC quantizer on a normalized 0..1 range."""
    n_codes = 2**n_bits
    clipped = np.clip(vin, 0.0, np.nextafter(1.0, 0.0))
    codes = np.floor(clipped * n_codes)
    return (codes + 0.5) / n_codes


def noise_rms_for_enob(amplitude: float, n_bits: int, enob: float) -> float:
    """Input noise needed after subtracting ideal quantization noise."""
    target_sndr = 6.02 * enob + 1.76
    signal_power = amplitude**2 / 2
    total_noise_power = signal_power / 10 ** (target_sndr / 10)
    quant_noise_power = (1 / 2**n_bits) ** 2 / 12
    return np.sqrt(max(total_noise_power - quant_noise_power, 0.0))


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": np.mean(values),
        "std": np.std(values, ddof=1),
        "min": np.min(values),
        "max": np.max(values),
        "pkpk": np.ptp(values),
    }


def plot_metric(n_ffts: np.ndarray, values: np.ndarray, ylabel: str, title: str, path: Path) -> None:
    means = np.mean(values, axis=1)
    stds = np.std(values, axis=1, ddof=1)
    mins = np.min(values, axis=1)
    maxs = np.max(values, axis=1)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.fill_between(n_ffts, mins, maxs, color="#9ecae1", alpha=0.32, label="MC min-max")
    ax.errorbar(
        n_ffts,
        means,
        yerr=stds,
        marker="o",
        ms=4,
        lw=1.6,
        capsize=3,
        color="#08519c",
        ecolor="#3182bd",
        label="mean +/- 1 sigma",
    )
    for x, samples in zip(n_ffts, values):
        ax.scatter(np.full(samples.shape, x), samples, s=9, color="#de2d26", alpha=0.32)

    ax.set_xscale("log", base=2)
    ax.set_xticks(n_ffts)
    ax.set_xticklabels([fr"$2^{{{int(np.log2(n))}}}$" for n in n_ffts], rotation=45)
    ax.set_xlabel("FFT points")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", color="#d9d9d9", lw=0.8, alpha=0.8)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


rng = np.random.default_rng(RNG_SEED)
noise_rms = noise_rms_for_enob(A, N_BITS, TARGET_ENOB)
hd3_amplitude = A * 10 ** (HD3_DBC / 20)
n_ffts = np.array([2**p for p in range(4, 17)])

rows = []
sfdr_mc = []
sndr_mc = []

print("=" * 80)
print("FFT LENGTH MONTE CARLO SWEEP: 10-bit ADC, ENOB=9 noise, HD3=-80 dBc")
print("=" * 80)
print(f"Monte Carlo runs per FFT length: {N_MC}")
print(f"Input noise RMS: {noise_rms:.6g} FS units")
print(f"HD3 amplitude: {hd3_amplitude:.6g} FS units ({HD3_DBC:.1f} dBc)")
print()

for n_fft in n_ffts:
    fin_bin = coherent_odd_bin(n_fft, FIN_RATIO)
    fin = fin_bin / n_fft * Fs
    n = np.arange(n_fft)
    clean = (
        DC
        + A * np.sin(2 * np.pi * fin_bin * n / n_fft)
        + hd3_amplitude * np.sin(2 * np.pi * 3 * fin_bin * n / n_fft)
    )

    sfdr_values = []
    sndr_values = []
    for _ in range(N_MC):
        aout = quantize_ideal(clean + rng.normal(0.0, noise_rms, size=n_fft), N_BITS)
        metrics = analyze_spectrum(aout, fs=Fs, create_plot=False, show_label=False)
        sfdr_values.append(metrics["sfdr_dbc"])
        sndr_values.append(metrics["sndr_dbc"])

    sfdr_values = np.array(sfdr_values)
    sndr_values = np.array(sndr_values)
    sfdr_mc.append(sfdr_values)
    sndr_mc.append(sndr_values)

    sfdr = summarize(sfdr_values)
    sndr = summarize(sndr_values)
    rows.append(
        {
            "n_fft": n_fft,
            "fin_bin": fin_bin,
            "fin_hz": fin,
            **{f"sfdr_{key}_dbc": value for key, value in sfdr.items()},
            **{f"sndr_{key}_dbc": value for key, value in sndr.items()},
        }
    )
    print(
        f"N={n_fft:6d}, bin={fin_bin:5d}, "
        f"SFDR={sfdr['mean']:6.2f} +/- {sfdr['std']:5.2f} dBc, "
        f"SNDR={sndr['mean']:6.2f} +/- {sndr['std']:5.2f} dBc"
    )

sfdr_mc = np.array(sfdr_mc)
sndr_mc = np.array(sndr_mc)

csv_path = output_dir / "exp_s13_fft_length_mc_sfdr_sndr.csv"
with csv_path.open("w", encoding="utf-8") as f:
    header = list(rows[0])
    f.write(",".join(header) + "\n")
    for row in rows:
        f.write(",".join(str(row[key]) for key in header) + "\n")

np.savez(
    output_dir / "exp_s13_fft_length_mc_sfdr_sndr_raw.npz",
    n_ffts=n_ffts,
    sfdr_mc=sfdr_mc,
    sndr_mc=sndr_mc,
    noise_rms=noise_rms,
    hd3_dbc=HD3_DBC,
    n_bits=N_BITS,
    target_enob=TARGET_ENOB,
    n_mc=N_MC,
    fs=Fs,
    rng_seed=RNG_SEED,
)

sfdr_fig_path = output_dir / "exp_s13_fft_length_mc_sfdr.png"
sndr_fig_path = output_dir / "exp_s13_fft_length_mc_sndr.png"
plot_metric(
    n_ffts,
    sfdr_mc,
    ylabel="SFDR (dBc)",
    title="10-bit ADC, ENOB=9 noise, HD3=-80 dBc: SFDR vs FFT length",
    path=sfdr_fig_path,
)
plot_metric(
    n_ffts,
    sndr_mc,
    ylabel="SNDR (dBc)",
    title="10-bit ADC, ENOB=9 noise, HD3=-80 dBc: SNDR vs FFT length",
    path=sndr_fig_path,
)

print(f"\n[Save csv] -> [{csv_path.resolve()}]")
print(f"[Save fig] -> [{sfdr_fig_path.resolve()}]")
print(f"[Save fig] -> [{sndr_fig_path.resolve()}]\n")
