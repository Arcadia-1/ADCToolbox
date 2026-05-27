"""MSB +1% SAR error: strict binary vs third-weight repeat calibration.

This example compares two 16-bit SAR weight lists under the same deterministic
MSB weight error:

1. Strict binary:
   ``[32768, 16384, 8192, 4096, ..., 1]``
2. Third-weight repeat:
   ``[32768, 16384, 8192, 8192, 4096, ..., 1]``

The actual CDAC has MSB +1%. The test first reconstructs with nominal weights
only, then calibrates weights from one coherent sine and evaluates the result
on a different coherent sine.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from adctoolbox import analyze_spectrum, calibrate_weight_sine
from adctoolbox.models import sar_convert, sar_reconstruct


output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)


NUM_BITS = 16
N_SAMPLES = 2**13
FS = float(N_SAMPLES)
TRAIN_BIN = 997
TEST_BIN = 1231
AMPLITUDE = 0.5
TEST_PHASE = 0.37
MSB_DELTA = 0.01


def strict_binary_raw_weights(num_bits: int) -> np.ndarray:
    """Return integer binary SAR weights, MSB first."""
    return 2.0 ** np.arange(num_bits - 1, -1, -1)


def third_weight_repeat_raw_weights(num_bits: int) -> np.ndarray:
    """Duplicate the third largest SAR weight to add redundancy."""
    raw = strict_binary_raw_weights(num_bits)
    return np.insert(raw, 3, raw[2])


def normalize_sar_weights(raw_weights: np.ndarray) -> np.ndarray:
    """Normalize SAR weights by sum(bit weights) + one LSB."""
    return raw_weights / (raw_weights.sum() + raw_weights[-1])


def first_decision_margin_lsb(raw_weights: np.ndarray, msb_delta: float) -> float:
    """Return remaining first-step correction margin in LSB units."""
    actual_msb = raw_weights[0] * (1.0 + msb_delta)
    return float(raw_weights[1:].sum() + raw_weights[-1] - actual_msb)


def centered(trace: np.ndarray) -> np.ndarray:
    """Remove DC before spectrum analysis."""
    return trace - np.mean(trace)


def calibrate_weights(bits: np.ndarray, nominal_weights: np.ndarray) -> np.ndarray:
    """Run sine calibration while keeping example output concise."""
    with contextlib.redirect_stdout(io.StringIO()):
        result = calibrate_weight_sine(
            bits,
            freq=TRAIN_BIN / N_SAMPLES,
            nominal_weights=nominal_weights,
        )
    # calibrate_weight_sine returns differential-scale weights for this
    # single-ended normalized SAR setup.
    return np.asarray(result["weight"], dtype=float) / 2.0


def analyze_trace(trace: np.ndarray, ax=None, title: str | None = None) -> dict:
    """Analyze one trace with the default ADCToolbox spectrum plotter."""
    metrics = analyze_spectrum(
        centered(trace),
        fs=FS,
        max_scale_range=(-0.5, 0.5),
        win_type="rectangular",
        side_bin=0,
        max_harmonic=5,
        nf_method=3,
        create_plot=ax is not None,
        ax=ax,
    )
    if ax is not None:
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlim(FS / N_SAMPLES, FS / 2)
        ax.set_ylim(-160, 5)
    return metrics


def main() -> None:
    n = np.arange(N_SAMPLES)
    vin_train = 0.5 + AMPLITUDE * np.sin(2 * np.pi * TRAIN_BIN * n / N_SAMPLES)
    vin_test = 0.5 + AMPLITUDE * np.sin(
        2 * np.pi * TEST_BIN * n / N_SAMPLES + TEST_PHASE
    )

    architectures = [
        ("Strict binary", strict_binary_raw_weights(NUM_BITS)),
        ("3rd-weight repeat", third_weight_repeat_raw_weights(NUM_BITS)),
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.8, 8.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    rows = []
    for col, (name, raw_weights) in enumerate(architectures):
        nominal_weights = normalize_sar_weights(raw_weights)
        actual_weights = nominal_weights.copy()
        actual_weights[0] *= 1.0 + MSB_DELTA

        bits_train = sar_convert(vin_train, actual_weights, quant_range=(0.0, 1.0))
        bits_test = sar_convert(vin_test, actual_weights, quant_range=(0.0, 1.0))

        before = sar_reconstruct(bits_test, nominal_weights, quant_range=(0.0, 1.0))
        calibrated_weights = calibrate_weights(bits_train, nominal_weights)
        after = bits_test.astype(float) @ calibrated_weights
        ideal_actual = sar_reconstruct(bits_test, actual_weights, quant_range=(0.0, 1.0))

        before_metrics = analyze_trace(before, axes[0, col], f"{name} - Before cal")
        after_metrics = analyze_trace(after, axes[1, col], f"{name} - After cal")
        ideal_metrics = analyze_trace(ideal_actual)

        rows.append(
            {
                "architecture": name,
                "msb_delta_pct": MSB_DELTA * 100.0,
                "n_weights": len(raw_weights),
                "nominal_sum_int": int(raw_weights.sum()),
                "first_decision_margin_lsb": first_decision_margin_lsb(
                    raw_weights,
                    MSB_DELTA,
                ),
                "before_enob": before_metrics["enob"],
                "after_enob": after_metrics["enob"],
                "ideal_actual_weight_enob": ideal_metrics["enob"],
                "before_sndr": before_metrics["sndr_dbc"],
                "after_sndr": after_metrics["sndr_dbc"],
                "ideal_actual_weight_sndr": ideal_metrics["sndr_dbc"],
            }
        )

    fig.suptitle(
        "16-bit SAR with MSB Actual Weight +1%\n"
        "Train calibration on one sine, evaluate on another sine",
        fontsize=14,
        fontweight="bold",
    )

    fig_path = output_dir / "exp_d17_sar_msb_error_binary_vs_repeat_calibration.png"
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"[Save fig] -> [{fig_path}]")
    print("architecture,first_margin_lsb,before_ENOB,after_ENOB,ideal_actual_ENOB")
    for row in rows:
        print(
            f"{row['architecture']},"
            f"{row['first_decision_margin_lsb']:.2f},"
            f"{row['before_enob']:.2f},"
            f"{row['after_enob']:.2f},"
            f"{row['ideal_actual_weight_enob']:.2f}"
        )


if __name__ == "__main__":
    main()
