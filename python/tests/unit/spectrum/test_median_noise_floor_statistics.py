"""Regression tests for median noise-floor statistics after spectrum averaging.

For white Gaussian noise, each complex FFT bin has exponentially distributed
power, so its median is ln(2) times its mean. Power averaging of M records
takes the mean of M independent exponentials (Gamma(M) / M). Coherent
averaging adds complex bins instead; the averaged bin stays complex Gaussian
and its power stays exponential for any number of records.
"""

import numpy as np
import pytest

from adctoolbox.spectrum import compute_spectrum
from adctoolbox.spectrum._spectrum_averaging import _coherent_average


TONE_BIN = 379
TONE_AMPLITUDE = 0.7
NOISE_RMS = 2e-3
SCALAR_METRICS = (
    "enob",
    "sndr_dbc",
    "sfdr_dbc",
    "snr_dbc",
    "sig_pwr_dbfs",
    "noise_floor_dbfs",
    "nsd_dbfs_hz",
    "thd_dbc",
)


def _noisy_tones(n_samples: int, n_records: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = np.arange(n_samples)
    phases = rng.uniform(-np.pi, np.pi, n_records)
    return np.array([
        TONE_AMPLITUDE * np.sin(2 * np.pi * TONE_BIN * n / n_samples + phase)
        + rng.normal(0, NOISE_RMS, n_samples)
        for phase in phases
    ])


def _spectrum(data: np.ndarray, coherent_averaging: bool, nf_method: int = 1) -> dict:
    return compute_spectrum(
        data,
        max_scale_range=[-1, 1],
        win_type="boxcar",
        side_bin=0,
        max_harmonic=5,
        nf_method=nf_method,
        coherent_averaging=coherent_averaging,
    )


@pytest.mark.parametrize("coherent_averaging", [False, True])
@pytest.mark.parametrize("n_records", [1, 2, 4, 16])
def test_median_noise_matches_direct_integration(coherent_averaging, n_records):
    # Coherent averaging aligns phases bin by bin in Python, so it uses a
    # shorter record.
    n_samples = 2**13 if coherent_averaging else 2**17
    data = _noisy_tones(n_samples, n_records, seed=20261103)

    noise_parts = _spectrum(data, coherent_averaging)["plot_data"]["noise_parts"]

    # Boxcar bins of white noise are independent. When each of n bins
    # averages K exponential powers (K = 1 coherent, K = M power averaging),
    # the sampling std of median/direct is below 1.04 / sqrt(K * n).
    n_exponentials = 1 if coherent_averaging else n_records
    tolerance = 4 * 1.04 / np.sqrt(n_exponentials * n_samples / 2)
    ratio = noise_parts["median"] / noise_parts["exclude"]
    assert ratio == pytest.approx(1.0, abs=tolerance)


@pytest.mark.parametrize("nf_method", [0, 1, 2, 3])
@pytest.mark.parametrize(
    "layout",
    [
        "DZZZZZZZ",  # One capture and seven empty records.
        "ZDZDDZ",  # Empty records before, between and after captures.
    ],
)
def test_coherent_average_ignores_skipped_empty_records(layout, nf_method):
    n_samples = 4096
    captures = _noisy_tones(n_samples, layout.count("D"), seed=20261104)
    rows = iter(captures)
    padded = np.array([
        next(rows) if record == "D" else np.zeros(n_samples) for record in layout
    ])

    reference = _spectrum(captures, coherent_averaging=True, nf_method=nf_method)
    result = _spectrum(padded, coherent_averaging=True, nf_method=nf_method)

    assert result["M"] == reference["M"] == layout.count("D")
    for key in SCALAR_METRICS:
        assert result["metrics"][key] == pytest.approx(
            reference["metrics"][key], rel=1e-12, abs=1e-12
        ), key
    np.testing.assert_allclose(
        result["metrics"]["harmonics_dbc"], reference["metrics"]["harmonics_dbc"],
        rtol=1e-12,
    )
    assert result["plot_data"]["noise_parts"] == pytest.approx(
        reference["plot_data"]["noise_parts"], rel=1e-12
    )


def test_coherent_average_reports_records_used():
    n_samples = 1024
    captures = _noisy_tones(n_samples, 3, seed=20261105)
    data = np.vstack([np.zeros(n_samples), captures[:2], np.zeros(n_samples), captures[2:]])

    _, _, n_averaged = _coherent_average(data, osr=1)

    assert n_averaged == 3


@pytest.mark.parametrize("style", ["standard", "virtuoso"])
@pytest.mark.parametrize("n_records", [1, 3])
def test_coherent_plot_labels_count_only_accepted_records(style, n_records):
    import matplotlib.pyplot as plt
    from adctoolbox.spectrum.plot_spectrum import plot_spectrum
    from adctoolbox.spectrum.plot_spectrum_virtuoso import plot_spectrum_virtuoso

    captures = _noisy_tones(1024, n_records, seed=20261109)
    padded = np.vstack([captures, np.zeros((7, 1024))])
    result = _spectrum(padded, coherent_averaging=True)
    plotter = plot_spectrum if style == "standard" else plot_spectrum_virtuoso
    fig, ax = plt.subplots()
    try:
        plotter(result, ax=ax)
        labels = "\n".join(text.get_text() for text in ax.texts)
        if n_records == 1:
            if style == "standard":
                assert ax.get_title() == "Coherent Spectrum"
            assert "Coherent Gain" not in labels
        else:
            if style == "standard":
                assert ax.get_title() == f"Coherent averaging (N_run = {n_records})"
            assert f"Coherent Gain = {10 * np.log10(n_records):.2f} dB" in labels
    finally:
        plt.close(fig)
