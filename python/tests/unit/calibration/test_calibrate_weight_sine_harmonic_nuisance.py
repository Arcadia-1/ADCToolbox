"""Regression tests for calibrate_weight_sine harmonic nuisance semantics."""

import numpy as np
import pytest

from adctoolbox.calibration import calibrate_weight_sine
from adctoolbox.spectrum import compute_spectrum


def _distorted_source_bits():
    """Create ideal ADC bits from a sine source with deterministic H3."""
    n_samples = 4096
    bit_width = 10
    fin_bin = 307
    fin = fin_bin / n_samples
    amplitude = 0.49
    h3_amplitude = 0.025

    n = np.arange(n_samples)
    signal = (
        amplitude * np.sin(2 * np.pi * fin * n)
        + h3_amplitude * np.sin(2 * np.pi * 3 * fin * n + 0.25)
    )

    levels = 2**bit_width - 1
    codes = np.round((signal + 1.0) * 0.5 * levels).astype(int)
    codes = np.clip(codes, 0, levels)
    bits = ((codes[:, None] >> np.arange(bit_width - 1, -1, -1)) & 1).astype(float)
    nominal_weights = 2.0 ** np.arange(bit_width - 1, -1, -1)
    return bits, nominal_weights, fin


def _spectrum_sndr_and_hd3(calibrated_signal):
    signal = np.squeeze(np.asarray(calibrated_signal, dtype=float))
    spectrum = compute_spectrum(
        signal,
        fs=1.0,
        max_scale_range=None,
        win_type="hann",
        side_bin=1,
        osr=1,
        max_harmonic=5,
        nf_method=3,
    )
    metrics = spectrum["metrics"]
    return metrics["sndr_dbc"], metrics["harmonics_dbc"][1]


def test_harmonic_order_fits_reference_harmonics_not_output_distortion():
    bits, nominal_weights, fin = _distorted_source_bits()

    fundamental_only = calibrate_weight_sine(
        bits,
        freq=fin,
        force_search=False,
        nominal_weights=nominal_weights,
        harmonic_order=1,
    )
    with_h3_nuisance = calibrate_weight_sine(
        bits,
        freq=fin,
        force_search=False,
        nominal_weights=nominal_weights,
        harmonic_order=3,
    )

    spectrum_sndr, spectrum_hd3 = _spectrum_sndr_and_hd3(
        with_h3_nuisance["calibrated_signal"]
    )

    assert with_h3_nuisance["snr_db"] > fundamental_only["snr_db"] + 20.0
    assert with_h3_nuisance["snr_db"] > spectrum_sndr + 20.0
    assert spectrum_hd3 == pytest.approx(-25.85, abs=1.0)
