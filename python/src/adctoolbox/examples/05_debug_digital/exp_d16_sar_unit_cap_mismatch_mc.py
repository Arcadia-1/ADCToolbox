"""SAR redundancy and foreground calibration under unit-cap mismatch.

This example compares two 16-bit SAR ADC weight lists:

1. Strict binary weights.
2. A redundant integer radix ~1.8 weight list with the same 16.00-bit span.

For each unit-cap mismatch sigma, the script runs 32 Monte Carlo samples using
Pelgrom-style capacitor mismatch, then compares ENOB before and after
foreground sine calibration.
"""

from __future__ import annotations

import contextlib
import csv
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from adctoolbox import (
    analyze_weight_radix,
    calibrate_weight_sine,
    quick_sndr,
    sar_apply_cap_mismatch,
    sar_convert,
    sar_ideal_weights,
    sar_reconstruct,
)


output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)


N_SAMPLES = 2**13
FS = 1.0
TRAIN_BIN = 997
TEST_BIN = 1231
AMPLITUDE = 0.499
N_MC = 32
SIGMA_PCT = np.arange(0.0, 10.0 + 0.25, 0.25)
BASE_SEED = 20260525


def radix18_integer_weights_16bit() -> np.ndarray:
    """Return normalized integer weights for a 16.00-bit radix ~1.8 SAR."""
    raw = np.array(
        [
            29127,
            16182,
            8990,
            4995,
            2775,
            1542,
            856,
            476,
            264,
            147,
            82,
            45,
            25,
            14,
            8,
            4,
            2,
            1,
        ],
        dtype=float,
    )
    return raw / (raw.sum() + raw[-1])


def enob_from_trace(aout: np.ndarray) -> float:
    """Compute ENOB with the lean spectrum path used for sweep loops."""
    centered = aout - np.mean(aout)
    return quick_sndr(centered, fs=FS, win_type="rectangular")["enob"]


def calibrate_weights(bits: np.ndarray, nominal_weights: np.ndarray) -> np.ndarray:
    """Run sine calibration while suppressing verbose solver diagnostics."""
    with contextlib.redirect_stdout(io.StringIO()):
        result = calibrate_weight_sine(
            bits,
            freq=TRAIN_BIN / N_SAMPLES,
            nominal_weights=nominal_weights,
        )
    return np.asarray(result["weight"], dtype=float)


def summarize(values: list[float]) -> dict[str, float]:
    """Summarize one Monte Carlo distribution."""
    data = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(data)),
        "p10": float(np.percentile(data, 10)),
        "q25": float(np.percentile(data, 25)),
        "median": float(np.median(data)),
        "q75": float(np.percentile(data, 75)),
        "p90": float(np.percentile(data, 90)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "std": float(np.std(data, ddof=1)) if len(data) > 1 else 0.0,
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    n = np.arange(N_SAMPLES)
    vin_train = 0.5 + AMPLITUDE * np.sin(2 * np.pi * TRAIN_BIN * n / N_SAMPLES)
    vin_test = 0.5 + AMPLITUDE * np.sin(
        2 * np.pi * TEST_BIN * n / N_SAMPLES + 0.37
    )

    architectures = [
        ("Strict binary", sar_ideal_weights(16), "#1f77b4"),
        ("Radix ~1.8", radix18_integer_weights_16bit(), "#d62728"),
    ]

    for name, weights, _color in architectures:
        info = analyze_weight_radix(weights, create_plot=False)
        print(f"[{name:13s}] effective span = {info['effres']:.2f} bits")

    raw_rows = []
    grouped: dict[tuple[str, str, float], list[float]] = {}

    for sigma_idx, sigma_pct in enumerate(SIGMA_PCT):
        sigma = sigma_pct / 100.0
        for trial in range(N_MC):
            for arch_idx, (arch_name, w_nominal, _color) in enumerate(architectures):
                seed = BASE_SEED + sigma_idx * 100_000 + trial * 100 + arch_idx
                rng = np.random.default_rng(seed)

                w_actual = sar_apply_cap_mismatch(w_nominal, sigma=sigma, rng=rng)
                bits_train = sar_convert(vin_train, w_actual)
                bits_test = sar_convert(vin_test, w_actual)

                before = sar_reconstruct(bits_test, w_nominal)
                before_enob = enob_from_trace(before)

                w_calibrated = calibrate_weights(bits_train, w_nominal)
                after = bits_test.astype(float) @ w_calibrated
                after_enob = enob_from_trace(after)

                for cal_state, enob in [
                    ("before cal", before_enob),
                    ("after cal", after_enob),
                ]:
                    raw_rows.append(
                        {
                            "sigma_pct": float(sigma_pct),
                            "trial": trial,
                            "architecture": arch_name,
                            "calibration": cal_state,
                            "enob": float(enob),
                        }
                    )
                    grouped.setdefault((arch_name, cal_state, float(sigma_pct)), []).append(
                        float(enob)
                    )

        if sigma_idx % 8 == 0 or sigma_idx == len(SIGMA_PCT) - 1:
            print(
                f"[Progress] sigma={sigma_pct:4.2f}% "
                f"({sigma_idx + 1}/{len(SIGMA_PCT)})"
            )

    stat_rows = []
    for (arch_name, cal_state, sigma_pct), values in sorted(grouped.items()):
        row = {
            "sigma_pct": sigma_pct,
            "architecture": arch_name,
            "calibration": cal_state,
            "n_mc": len(values),
        }
        row.update(summarize(values))
        stat_rows.append(row)

    raw_csv = output_dir / "exp_d16_sar_unit_cap_mismatch_mc_raw.csv"
    stats_csv = output_dir / "exp_d16_sar_unit_cap_mismatch_mc_stats.csv"
    write_csv(
        raw_csv,
        raw_rows,
        ["sigma_pct", "trial", "architecture", "calibration", "enob"],
    )
    write_csv(
        stats_csv,
        stat_rows,
        [
            "sigma_pct",
            "architecture",
            "calibration",
            "n_mc",
            "min",
            "p10",
            "q25",
            "median",
            "q75",
            "p90",
            "max",
            "mean",
            "std",
        ],
    )

    fig, ax = plt.subplots(figsize=(8.2, 4.9), constrained_layout=True)
    styles = [
        ("Strict binary", "before cal", "#1f77b4", "--"),
        ("Strict binary", "after cal", "#1f77b4", "-"),
        ("Radix ~1.8", "before cal", "#d62728", "--"),
        ("Radix ~1.8", "after cal", "#d62728", "-"),
    ]

    for arch_name, cal_state, color, line_style in styles:
        rows = [
            row
            for row in stat_rows
            if row["architecture"] == arch_name and row["calibration"] == cal_state
        ]
        rows.sort(key=lambda row: row["sigma_pct"])

        x = np.array([row["sigma_pct"] for row in rows])
        y_min = np.array([row["min"] for row in rows])
        y_p10 = np.array([row["p10"] for row in rows])
        y_med = np.array([row["median"] for row in rows])
        y_p90 = np.array([row["p90"] for row in rows])
        y_max = np.array([row["max"] for row in rows])

        alpha_range = 0.055 if cal_state == "before cal" else 0.075
        alpha_p1090 = 0.12 if cal_state == "before cal" else 0.16
        ax.fill_between(x, y_min, y_max, color=color, alpha=alpha_range, linewidth=0)
        ax.fill_between(x, y_p10, y_p90, color=color, alpha=alpha_p1090, linewidth=0)
        ax.plot(
            x,
            y_med,
            color=color,
            linestyle=line_style,
            linewidth=2.25,
            label=f"{arch_name}, {cal_state}",
        )

    ax.axhline(16.0, color="#555555", lw=0.9, ls=":", alpha=0.8)
    ax.set_title(
        "16-bit SAR Pelgrom/unit-cap mismatch, 32-run ENOB distribution",
        fontsize=13,
    )
    ax.set_xlabel("Unit-cap mismatch sigma (%)")
    ax.set_ylabel("ENOB from quick_sndr (bits)")
    ax.set_xlim(0, 10)
    ax.set_ylim(10.0, 16.35)
    ax.set_xticks(np.arange(0, 10.1, 1.0))
    ax.grid(True, color="#d9d9d9", linewidth=0.75, linestyle="--", alpha=0.75)
    ax.text(
        0.06,
        0.84,
        "After cal.",
        transform=ax.transAxes,
        fontsize=22,
        fontweight="bold",
        color="#222222",
        alpha=0.78,
    )
    ax.text(
        0.56,
        0.13,
        "Before cal.",
        transform=ax.transAxes,
        fontsize=22,
        fontweight="bold",
        color="#222222",
        alpha=0.78,
    )
    ax.legend(loc="lower left", ncol=2, fontsize=8.8, frameon=True)

    fig_path = output_dir / "exp_d16_sar_unit_cap_mismatch_mc.png"
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"[Save fig] -> [{fig_path}]")
    print(f"[Save raw CSV] -> [{raw_csv}]")
    print(f"[Save stats CSV] -> [{stats_csv}]")


if __name__ == "__main__":
    main()
