"""Training-length sweep for redundant SAR calibration under cap mismatch.

This example uses the same redundant 16-bit SAR weight list and
Pelgrom/unit-cap mismatch model as ``exp_d16_sar_unit_cap_mismatch_mc.py``.
It asks a different question: how many coherent training samples are needed
before foreground sine calibration generalizes to a fixed 16384-sample test
capture?

For each training length from ``2**4`` to ``2**14``, the script runs 32
Monte Carlo trials. Each trial uses a different mismatch realization and
different sine starting phases. The main plot shows the calibrated ENOB
distribution envelope on an independent test capture versus training length.

The second plot intentionally evaluates the same calibrated weights on the
calibration capture itself, using the same visual style as the main plot.
Comparing the two saved figures exposes the overfitting case: short training
records can look excellent on the data used to solve the weights while failing
to generalize.
"""

from __future__ import annotations

import contextlib
import csv
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from adctoolbox import (
    calibrate_weight_sine,
    quick_sndr,
    sar_apply_cap_mismatch,
    sar_convert,
)


output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)


N_TEST = 2**14
FS = 1.0
TEST_BIN = 1777
TRAIN_TARGET_BIN_AT_N_TEST = 997
TRAIN_LENGTHS = 2 ** np.arange(4, 15)
AMPLITUDE = 0.499
N_MC = 32
MISMATCH_SIGMA_PCT = 1.0
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


def coherent_train_bin(n_samples: int) -> int:
    """Pick an odd coherent training bin near the target normalized tone."""
    target_ratio = TRAIN_TARGET_BIN_AT_N_TEST / N_TEST
    bin_index = int(round(target_ratio * n_samples))
    bin_index = max(1, min(bin_index, n_samples // 2 - 1))
    if bin_index % 2 == 0:
        if bin_index + 1 < n_samples // 2:
            bin_index += 1
        elif bin_index > 1:
            bin_index -= 1
    return bin_index


def sine_capture(n_samples: int, bin_index: int, phase: float) -> np.ndarray:
    """Generate a coherent normalized SAR input sine."""
    n = np.arange(n_samples)
    return 0.5 + AMPLITUDE * np.sin(2 * np.pi * bin_index * n / n_samples + phase)


SUMMARY_KEYS = [
    "n_valid",
    "n_fail",
    "min",
    "p10",
    "q25",
    "median",
    "q75",
    "p90",
    "max",
    "mean",
    "std",
]


def spectrum_metrics(trace: np.ndarray) -> dict[str, float]:
    """Compute FFT SNDR/ENOB for a reconstructed trace."""
    centered = trace - np.mean(trace)
    return quick_sndr(centered, fs=FS, win_type="rectangular")


def calibrate_weights(
    bits: np.ndarray,
    train_bin: int,
    n_train: int,
    nominal_weights: np.ndarray,
) -> np.ndarray:
    """Run sine calibration while suppressing solver diagnostics."""
    with contextlib.redirect_stdout(io.StringIO()):
        result = calibrate_weight_sine(
            bits,
            freq=train_bin / n_train,
            nominal_weights=nominal_weights,
        )
    return np.asarray(result["weight"], dtype=float)


def summarize(values: list[float]) -> dict[str, float | int]:
    """Summarize one Monte Carlo distribution, preserving failed runs."""
    data = np.asarray(values, dtype=float)
    finite = data[np.isfinite(data)]
    row: dict[str, float | int] = {
        "n_valid": int(len(finite)),
        "n_fail": int(len(data) - len(finite)),
    }
    if len(finite) == 0:
        row.update(
            {
                "min": np.nan,
                "p10": np.nan,
                "q25": np.nan,
                "median": np.nan,
                "q75": np.nan,
                "p90": np.nan,
                "max": np.nan,
                "mean": np.nan,
                "std": np.nan,
            }
        )
        return row

    row.update(
        {
            "min": float(np.min(finite)),
            "p10": float(np.percentile(finite, 10)),
            "q25": float(np.percentile(finite, 25)),
            "median": float(np.median(finite)),
            "q75": float(np.percentile(finite, 75)),
            "p90": float(np.percentile(finite, 90)),
            "max": float(np.max(finite)),
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
        }
    )
    return row


def prefixed_summary(prefix: str, row: dict[str, float | int]) -> dict[str, float | int]:
    """Copy summary keys with a prefix while preserving the legacy test keys."""
    return {f"{prefix}_{key}": row[key] for key in SUMMARY_KEYS}


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_distribution_envelope(
    ax,
    x: np.ndarray,
    rows: list[dict],
    prefix: str,
    color: str,
    title: str,
    ylabel: str,
    y_top: float | None = None,
) -> None:
    """Plot the min/max, percentile bands, and median for one ENOB distribution."""

    def key(name: str) -> str:
        return name if prefix == "" else f"{prefix}_{name}"

    n_valid = np.array([row[key("n_valid")] for row in rows], dtype=int)
    y_min = np.array([row[key("min")] for row in rows], dtype=float)
    y_p10 = np.array([row[key("p10")] for row in rows], dtype=float)
    y_q25 = np.array([row[key("q25")] for row in rows], dtype=float)
    y_med = np.array([row[key("median")] for row in rows], dtype=float)
    y_q75 = np.array([row[key("q75")] for row in rows], dtype=float)
    y_p90 = np.array([row[key("p90")] for row in rows], dtype=float)
    y_max = np.array([row[key("max")] for row in rows], dtype=float)

    finite_mask = n_valid > 0
    if not np.any(finite_mask):
        return

    ax.fill_between(
        x[finite_mask],
        y_min[finite_mask],
        y_max[finite_mask],
        color=color,
        alpha=0.10,
        linewidth=0,
        label="min-max (all valid runs)",
    )
    ax.fill_between(
        x[finite_mask],
        y_p10[finite_mask],
        y_p90[finite_mask],
        color=color,
        alpha=0.18,
        linewidth=0,
        label="P10-P90 (middle 80%)",
    )
    ax.fill_between(
        x[finite_mask],
        y_q25[finite_mask],
        y_q75[finite_mask],
        color=color,
        alpha=0.28,
        linewidth=0,
        label="Q25-Q75 (middle 50%)",
    )
    ax.plot(
        x[finite_mask],
        y_med[finite_mask],
        color=color,
        linewidth=2.2,
        marker="o",
        label="median ENOB",
    )

    failed_mask = ~finite_mask
    if np.any(failed_mask):
        marker_y = np.nanmin(y_min[finite_mask]) - 0.08
        ax.scatter(
            x[failed_mask],
            np.full(np.sum(failed_mask), marker_y),
            marker="x",
            color="#d62728",
            label="all 32 runs failed",
        )

    ax.axhline(16.0, color="#555555", lw=0.9, ls=":", alpha=0.85)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in x], rotation=35, ha="right")
    ax.set_xlim(x[0] * 0.85, x[-1] * 1.15)
    if y_top is None:
        y_top = np.nanmax(y_max[finite_mask]) + 0.10
    ax.set_ylim(max(0.0, np.nanmin(y_min[finite_mask]) - 0.25), y_top)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Training samples")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", linestyle="--", linewidth=0.75, alpha=0.70)
    ax.legend(title="32-run ENOB distribution", loc="lower right", frameon=True)


def main() -> None:
    nominal_weights = radix18_integer_weights_16bit()
    sigma = MISMATCH_SIGMA_PCT / 100.0

    raw_rows = []
    stat_rows = []
    grouped_test_enob: dict[int, list[float]] = {int(n): [] for n in TRAIN_LENGTHS}
    grouped_train_enob: dict[int, list[float]] = {int(n): [] for n in TRAIN_LENGTHS}
    grouped_test_sndr: dict[int, list[float]] = {int(n): [] for n in TRAIN_LENGTHS}
    grouped_train_sndr: dict[int, list[float]] = {int(n): [] for n in TRAIN_LENGTHS}

    for trial in range(N_MC):
        chip_rng = np.random.default_rng(BASE_SEED + trial)
        actual_weights = sar_apply_cap_mismatch(
            nominal_weights,
            sigma=sigma,
            rng=chip_rng,
        )
        test_phase = chip_rng.uniform(0, 2 * np.pi)
        vin_test = sine_capture(N_TEST, TEST_BIN, test_phase)
        bits_test = sar_convert(vin_test, actual_weights)

        for n_train in TRAIN_LENGTHS:
            n_train = int(n_train)
            train_bin = coherent_train_bin(n_train)
            phase_rng = np.random.default_rng(BASE_SEED + 10_000_000 + trial * 1000 + n_train)
            train_phase = phase_rng.uniform(0, 2 * np.pi)
            vin_train = sine_capture(n_train, train_bin, train_phase)
            bits_train = sar_convert(vin_train, actual_weights)

            try:
                calibrated_weights = calibrate_weights(
                    bits_train,
                    train_bin,
                    n_train,
                    nominal_weights,
                )
                calibrated_train_trace = bits_train.astype(float) @ calibrated_weights
                calibrated_test_trace = bits_test.astype(float) @ calibrated_weights
                train_metrics = spectrum_metrics(calibrated_train_trace)
                test_metrics = spectrum_metrics(calibrated_test_trace)
                train_enob = float(train_metrics["enob"])
                test_enob = float(test_metrics["enob"])
                train_sndr = float(train_metrics["sndr_dbc"])
                test_sndr = float(test_metrics["sndr_dbc"])
                status = "ok"
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                train_enob = np.nan
                test_enob = np.nan
                train_sndr = np.nan
                test_sndr = np.nan
                status = type(exc).__name__

            grouped_test_enob[n_train].append(test_enob)
            grouped_train_enob[n_train].append(train_enob)
            grouped_test_sndr[n_train].append(test_sndr)
            grouped_train_sndr[n_train].append(train_sndr)
            raw_rows.append(
                {
                    "n_train": n_train,
                    "train_bin": train_bin,
                    "trial": trial,
                    "mismatch_sigma_pct": MISMATCH_SIGMA_PCT,
                    "train_phase_rad": train_phase,
                    "test_phase_rad": test_phase,
                    "train_enob": train_enob,
                    "test_enob": test_enob,
                    "train_sndr_db": train_sndr,
                    "test_sndr_db": test_sndr,
                    "calibrated_enob": test_enob,
                    "status": status,
                }
            )

        print(f"[Progress] trial {trial + 1}/{N_MC}")

    for n_train in TRAIN_LENGTHS:
        n_train = int(n_train)
        row = {
            "n_train": n_train,
            "train_bin": coherent_train_bin(n_train),
            "mismatch_sigma_pct": MISMATCH_SIGMA_PCT,
            "n_mc": N_MC,
        }
        test_enob_summary = summarize(grouped_test_enob[n_train])
        row.update(test_enob_summary)
        row.update(prefixed_summary("test_enob", test_enob_summary))
        row.update(prefixed_summary("train_enob", summarize(grouped_train_enob[n_train])))
        row.update(prefixed_summary("test_sndr", summarize(grouped_test_sndr[n_train])))
        row.update(prefixed_summary("train_sndr", summarize(grouped_train_sndr[n_train])))
        stat_rows.append(row)

    raw_csv = output_dir / "exp_d18_sar_redundant_mismatch_training_length_sweep_raw.csv"
    stats_csv = output_dir / "exp_d18_sar_redundant_mismatch_training_length_sweep_stats.csv"
    write_csv(
        raw_csv,
        raw_rows,
        [
            "n_train",
            "train_bin",
            "trial",
            "mismatch_sigma_pct",
            "train_phase_rad",
            "test_phase_rad",
            "train_enob",
            "test_enob",
            "train_sndr_db",
            "test_sndr_db",
            "calibrated_enob",
            "status",
        ],
    )
    write_csv(
        stats_csv,
        stat_rows,
        [
            "n_train",
            "train_bin",
            "mismatch_sigma_pct",
            "n_mc",
            "n_valid",
            "n_fail",
            "min",
            "p10",
            "q25",
            "median",
            "q75",
            "p90",
            "max",
            "mean",
            "std",
            *[f"test_enob_{key}" for key in SUMMARY_KEYS],
            *[f"train_enob_{key}" for key in SUMMARY_KEYS],
            *[f"test_sndr_{key}" for key in SUMMARY_KEYS],
            *[f"train_sndr_{key}" for key in SUMMARY_KEYS],
        ],
    )

    x = np.array([row["n_train"] for row in stat_rows], dtype=float)
    y_max = np.array([row["max"] for row in stat_rows], dtype=float)
    n_valid = np.array([row["n_valid"] for row in stat_rows], dtype=int)

    test_title = (
        "Redundant 16-bit SAR calibration vs training length\n"
        f"{N_MC} runs, {MISMATCH_SIGMA_PCT:.1f}% unit-cap mismatch, test N={N_TEST}"
    )
    fig, ax = plt.subplots(figsize=(8.6, 5.1), constrained_layout=True)
    plot_distribution_envelope(
        ax,
        x,
        stat_rows,
        prefix="",
        color="#1f77b4",
        title=test_title,
        ylabel="Calibrated ENOB on 16384-sample test capture",
        y_top=min(16.35, np.nanmax(y_max[n_valid > 0]) + 0.10),
    )

    fig_path = output_dir / "exp_d18_sar_redundant_mismatch_training_length_sweep.png"
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    train_y_max = np.array([row["train_enob_max"] for row in stat_rows], dtype=float)
    train_valid = np.array([row["train_enob_n_valid"] for row in stat_rows], dtype=int) > 0
    train_title = (
        "Redundant 16-bit SAR calibration vs training length\n"
        f"{N_MC} runs, {MISMATCH_SIGMA_PCT:.1f}% unit-cap mismatch, calibration capture"
    )
    fig, ax = plt.subplots(figsize=(8.6, 5.1), constrained_layout=True)
    plot_distribution_envelope(
        ax,
        x,
        stat_rows,
        prefix="train_enob",
        color="#ff7f0e",
        title=train_title,
        ylabel="Calibrated ENOB on calibration capture",
        y_top=min(17.35, np.nanmax(train_y_max[train_valid]) + 0.10),
    )

    fit_fig_path = output_dir / "exp_d18_sar_redundant_mismatch_training_capture_overfit.png"
    fig.savefig(fit_fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"[Save fig] -> [{fig_path}]")
    print(f"[Save overfit fig] -> [{fit_fig_path}]")
    print(f"[Save raw CSV] -> [{raw_csv}]")
    print(f"[Save stats CSV] -> [{stats_csv}]")


if __name__ == "__main__":
    main()
