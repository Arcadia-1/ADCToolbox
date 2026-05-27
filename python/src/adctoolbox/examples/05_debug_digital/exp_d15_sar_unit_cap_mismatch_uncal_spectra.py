"""16-bit SAR unit-cap mismatch spectra without digital calibration.

This example intentionally keeps the analog CDAC weights and digital
reconstruction weights separate:

1. ``sar_convert`` uses a mismatched actual CDAC.
2. ``sar_reconstruct`` uses the ideal nominal binary weights.

That models an uncalibrated SAR backend and makes the harmonic/spur penalty
from capacitor mismatch visible in a four-panel spectrum comparison.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from adctoolbox import analyze_spectrum
from adctoolbox.models import (
    sar_apply_cap_mismatch,
    sar_convert,
    sar_ideal_weights,
    sar_reconstruct,
)


output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)


NUM_BITS = 16
N_SAMPLES = 16384
FS = 16384.0
FIN_BIN = 1777
AMPLITUDE = 0.5
BASE_SEED = 20260525

SIGMA_CASES = [
    (0.0, "0%"),
    (0.001, "0.1%"),
    (0.01, "1%"),
    (0.10, "10%"),
]


def main() -> None:
    nominal_weights = sar_ideal_weights(NUM_BITS)
    n = np.arange(N_SAMPLES)
    vin = 0.5 + AMPLITUDE * np.sin(2 * np.pi * FIN_BIN * n / N_SAMPLES)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 8.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = axes.ravel()
    metric_rows = []

    for idx, (sigma, label) in enumerate(SIGMA_CASES):
        rng = np.random.default_rng(BASE_SEED + idx)
        actual_weights = sar_apply_cap_mismatch(nominal_weights, sigma=sigma, rng=rng)

        bits = sar_convert(vin, actual_weights, quant_range=(0.0, 1.0))
        aout = sar_reconstruct(bits, nominal_weights, quant_range=(0.0, 1.0)) - 0.5

        ax = axes[idx]
        metrics = analyze_spectrum(
            data=aout,
            fs=FS,
            max_scale_range=(-0.5, 0.5),
            win_type="rectangular",
            side_bin=0,
            max_harmonic=5,
            nf_method=3,
            create_plot=True,
            ax=ax,
        )

        ax.set_title(f"sigma_Cu = {label}", fontsize=11, fontweight="bold")
        ax.set_xlim(FS / N_SAMPLES, FS / 2)
        ax.set_ylim(-180, 5)

        metric_rows.append(
            (
                label,
                metrics["sndr_dbc"],
                metrics["enob"],
                metrics["sfdr_dbc"],
                metrics["thd_dbc"],
            )
        )

    fig.suptitle(
        "16-bit Strict Binary SAR ADC with Unit-Cap Mismatch\n"
        "Encode with mismatched CDAC, reconstruct with nominal weights only; "
        f"Fin/Fs = {FIN_BIN}/{N_SAMPLES}",
        fontsize=14,
        fontweight="bold",
    )

    fig_path = output_dir / "exp_d15_sar_unit_cap_mismatch_uncal_spectra.png"
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"[Save fig] -> [{fig_path}]")
    print("sigma_Cu,SNDR_dB,ENOB,SFDR_dB,THD_dB")
    for label, sndr, enob, sfdr, thd in metric_rows:
        print(f"{label},{sndr:.2f},{enob:.2f},{sfdr:.2f},{thd:.2f}")


if __name__ == "__main__":
    main()
