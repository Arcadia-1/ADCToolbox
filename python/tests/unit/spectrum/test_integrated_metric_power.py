"""Regression tests for integrated-lobe spectrum metric definitions."""

import numpy as np
import pytest

from adctoolbox.spectrum.compute_spectrum import compute_spectrum


def _tone_with_hd2(n_samples=8192, fundamental_bin=123.0, hd2_dbc=-60.0):
    n = np.arange(n_samples)
    fundamental = np.sin(2 * np.pi * fundamental_bin * n / n_samples)
    hd2 = 10 ** (hd2_dbc / 20) * np.sin(2 * np.pi * 2 * fundamental_bin * n / n_samples)
    return fundamental + hd2


@pytest.mark.parametrize(
    "win_type,side_bin",
    [
        ("boxcar", 0),
        ("hann", 3),
        ("blackmanharris", 5),
        ("flattop", 6),
    ],
)
def test_windowed_coherent_metrics_use_integrated_lobe_power(win_type, side_bin):
    signal = _tone_with_hd2(fundamental_bin=123.0, hd2_dbc=-60.0)

    result = compute_spectrum(
        signal,
        fs=1.0,
        max_scale_range=[-1, 1],
        win_type=win_type,
        side_bin=side_bin,
        max_harmonic=5,
        assumed_sig_pwr_dbfs=0.0,
    )
    metrics = result["metrics"]

    assert metrics["sig_pwr_dbfs"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["harmonics_dbc"][0] == pytest.approx(-60.0, abs=1e-6)
    assert metrics["thd_dbc"] == pytest.approx(-60.0, abs=1e-6)
    assert metrics["sfdr_dbc"] == pytest.approx(60.0, abs=1e-6)


@pytest.mark.parametrize(
    "win_type,side_bin",
    [
        ("hann", 3),
        ("blackmanharris", 5),
        ("flattop", 6),
    ],
)
def test_noncoherent_thd_uses_integrated_harmonic_lobes(win_type, side_bin):
    signal = _tone_with_hd2(fundamental_bin=123.37, hd2_dbc=-60.0)

    result = compute_spectrum(
        signal,
        fs=1.0,
        max_scale_range=[-1, 1],
        win_type=win_type,
        side_bin=side_bin,
        max_harmonic=5,
    )
    metrics = result["metrics"]

    assert metrics["harmonics_dbc"][0] == pytest.approx(-60.0, abs=2e-3)
    assert metrics["thd_dbc"] == pytest.approx(-60.0, abs=2e-3)
