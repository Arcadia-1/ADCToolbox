"""Full-band versus in-band weight fitting on a noise-shaping SAR.

An ideal second-order error-feedback loop surrounds the public SAR model:
v[n] = x[n] - 2*e[n-1] + e[n-2], q[n] = SAR(v[n]), e[n] = q[n]-v[n].
Thus q-x = (1-z^-1)^2 e. The SAR uses mismatched analog DAC weights;
digital reconstruction initially uses nominal weights. This behavioral loop
has ideal residue storage/feedback, not a transistor-level noise-shaping SAR.

Fit on one capture, freeze weights, and measure a different frequency,
amplitude and phase. Every method is scored with the same in-band FFT mask.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from adctoolbox import calibrate_weight_sine, scale_calibration_output
from adctoolbox.models import sar_convert, sar_ideal_weights
from adctoolbox.spectrum import compute_spectrum


def simulate_noise_shaping_sar(inputs, analog_weights):
    """Sequential error feedback; parallel independent captures in columns."""
    rng = np.random.default_rng(91)
    decisions = np.empty((*inputs.shape, len(analog_weights)), dtype=np.int8)
    error1 = np.zeros(inputs.shape[1])
    error2 = np.zeros_like(error1)
    overloads = 0
    for n, signal in enumerate(inputs):
        quantizer_input = signal - 2 * error1 + error2
        overloads += np.count_nonzero((quantizer_input < 0) | (quantizer_input > 1))
        decisions[n] = sar_convert(quantizer_input, analog_weights, rng=rng)
        # Center the floor SAR's quantization error for the feedback loop.
        quantized = decisions[n] @ analog_weights + analog_weights[-1] / 2
        error2, error1 = error1, quantized - quantizer_input
    return decisions, int(overloads)


def run_comparison(n_samples=8192, osr=16):
    nominal = sar_ideal_weights(6)
    analog = nominal * np.array([1.03, 0.98, 1.04, 0.94, 1.08, 0.90])
    frequencies = np.array([23, 31]) / n_samples
    if frequencies.max() >= 0.5 / osr:
        raise ValueError("Use enough samples to keep both demonstration tones in band.")
    burn = 512
    t = np.arange(n_samples + burn) - burn
    inputs = 0.5 + np.array([0.33, 0.28]) * np.sin(
        2 * np.pi * t[:, None] * frequencies + np.array([0.21, 1.3]))
    decisions, overloads = simulate_noise_shaping_sar(inputs, analog)
    if overloads:
        raise RuntimeError(f"Demonstration overloaded on {overloads} samples.")
    train, test = decisions[burn:, 0], decisions[burn:, 1]

    weights = {"Uncalibrated": nominal}
    for label, fit_osr in [("Full-band fit", 1), ("In-band fit", osr)]:
        fit = calibrate_weight_sine(train, freq=frequencies[0],
                                    nominal_weights=nominal, osr=fit_osr)
        # Both fits use only the nominal scale, never the true analog weights.
        weights[label] = scale_calibration_output(fit, target_weights=nominal)["weight"]
    weights["Analog-weight reference"] = analog * nominal.sum() / analog.sum()

    spectra = {}
    for label, weight in weights.items():
        spectra[label] = compute_spectrum(test @ weight, fs=1, osr=osr,
                                          win_type="hann", side_bin=1,
                                          max_scale_range=(0, 1), nf_method=3)
    report = {
        "architecture": "6-bit SAR, ideal second-order residue error feedback",
        "osr": osr,
        "train_samples": n_samples,
        "test_samples": n_samples,
        "train_frequency_normalized": float(frequencies[0]),
        "test_frequency_normalized": float(frequencies[1]),
        "overload_samples": overloads,
        "metrics": {label: {name: float(result["metrics"][name])
                             for name in ["sndr_dbc", "sfdr_dbc", "enob"]}
                    for label, result in spectra.items()},
        "digital_weights": {label: values.tolist() for label, values in weights.items()},
        "analog_weights": analog.tolist(),
        "note": "All metrics use the held-out capture and the same in-band FFT mask. "
                "The analog-weight reference is not a bound on optimal digital correction.",
    }
    return report, spectra


def save_figures(report, spectra, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = ["#888888", "#D55E00", "#0072B2", "#009E73"]
    cutoff = 0.5 / report["osr"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5), layout="constrained")
    for ax in axes:
        for (label, result), color in zip(spectra.items(), colors):
            data = result["plot_data"]
            sndr = result["metrics"]["sndr_dbc"]
            ax.plot(data["freq"], data["power_spectrum_db_plot"], color=color,
                    lw=1.2, alpha=0.9, label=f"{label}: {sndr:.1f} dB in-band SNDR")
        ax.set_ylim(-145, 5)
        ax.set_ylabel("Power (dBFS/bin)", fontsize=12)
        ax.set_xlabel("Frequency / sampling rate", fontsize=12)
        ax.grid(alpha=0.2)
    axes[0].set_xlim(0, 0.5)
    axes[0].axvspan(0, cutoff, color="#F6D743", alpha=0.3)
    axes[0].set_title("Full spectrum — yellow area is the signal band", fontsize=13)
    axes[0].legend(fontsize=10, loc="upper right")
    axes[1].set_xlim(0, cutoff)
    axes[1].set_title("Same held-out capture — signal-band detail", fontsize=13)
    fig.suptitle("Oversampled SAR calibration", fontsize=16)
    fig.savefig(output_dir / "inband_weight_calibration.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    reference = np.array(report["digital_weights"]["Analog-weight reference"])
    for (label, values), color in zip(report["digital_weights"].items(), colors):
        if label == "Analog-weight reference":
            continue
        ax.plot(np.arange(6), 100 * (np.array(values) / reference - 1),
                "o-", color=color, lw=1.8, label=label)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(np.arange(6), ["MSB", "Bit 4", "Bit 3", "Bit 2", "Bit 1", "LSB"])
    ax.set_ylabel("Weight error relative to analog weights (%)", fontsize=12)
    ax.set_title("Same ADC — different calibration objectives", fontsize=14)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(output_dir / "inband_weight_errors.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output")
    args = parser.parse_args()
    report, spectra = run_comparison()
    save_figures(report, spectra, args.output)
    (args.output / "inband_weight_calibration.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for label, metrics in report["metrics"].items():
        print(f"{label:26s} SNDR={metrics['sndr_dbc']:7.2f} dB  ENOB={metrics['enob']:5.2f}")
    print(f"Saved figures and metrics to {args.output.resolve()}")


if __name__ == "__main__":
    main()
