"""Regression tests for dynamic metric mask consistency."""

import numpy as np
import pytest

from adctoolbox.spectrum._estimate_noise_power import _estimate_noise_power
from adctoolbox.spectrum.compute_spectrum import compute_spectrum


def _coherent_tone(n_samples: int, bin_index: int, amplitude: float = 1.0) -> np.ndarray:
    n = np.arange(n_samples)
    return amplitude * np.sin(2 * np.pi * bin_index * n / n_samples)


def test_nf_method3_excludes_windowed_harmonic_lobes_from_snr():
    n_samples = 8192
    fundamental_bin = 257
    hd2_dbc = -60.0
    noise_dbc = -80.0
    noise_bin = 1400

    signal = (
        _coherent_tone(n_samples, fundamental_bin)
        + _coherent_tone(n_samples, 2 * fundamental_bin, 10 ** (hd2_dbc / 20))
        + _coherent_tone(n_samples, noise_bin, 10 ** (noise_dbc / 20))
    )

    result = compute_spectrum(
        signal,
        fs=1.0,
        max_scale_range=[-1, 1],
        win_type="hann",
        side_bin=1,
        max_harmonic=5,
        nf_method=3,
        assumed_sig_pwr_dbfs=0.0,
    )
    metrics = result["metrics"]

    expected_sndr = 10 * np.log10(1 / (10 ** (hd2_dbc / 10) + 10 ** (noise_dbc / 10)))

    assert metrics["snr_dbc"] == pytest.approx(80.0, abs=1e-9)
    assert metrics["harmonics_dbc"][0] == pytest.approx(hd2_dbc, abs=1e-9)
    assert metrics["thd_dbc"] == pytest.approx(hd2_dbc, abs=1e-9)
    assert metrics["sndr_dbc"] == pytest.approx(expected_sndr, abs=1e-9)


def test_nf_method3_does_not_change_sndr_or_enob():
    n_samples = 8192
    signal = (
        _coherent_tone(n_samples, 257)
        + _coherent_tone(n_samples, 514, 10 ** (-60 / 20))
        + _coherent_tone(n_samples, 1400, 10 ** (-80 / 20))
    )

    median_result = compute_spectrum(
        signal,
        fs=1.0,
        max_scale_range=[-1, 1],
        win_type="hann",
        side_bin=1,
        max_harmonic=5,
        nf_method=1,
        assumed_sig_pwr_dbfs=0.0,
    )
    exclude_result = compute_spectrum(
        signal,
        fs=1.0,
        max_scale_range=[-1, 1],
        win_type="hann",
        side_bin=1,
        max_harmonic=5,
        nf_method=3,
        assumed_sig_pwr_dbfs=0.0,
    )

    assert exclude_result["metrics"]["sndr_dbc"] == median_result["metrics"]["sndr_dbc"]
    assert exclude_result["metrics"]["enob"] == median_result["metrics"]["enob"]


def test_nf_method3_preserves_dc_collision_boundary():
    spectrum_power = np.zeros(16)
    spectrum_power[3] = 1.0
    spectrum_power[4] = 2.0
    spectrum_power[12] = 4.0

    noise_power = _estimate_noise_power(
        spectrum_power=spectrum_power,
        nf_method=3,
        n_inband=16,
        M=1,
        bin_idx=8,
        harmonic_bins=np.array([2]),
        side_bin=2,
    )

    assert noise_power == pytest.approx(7.0)


def test_osr_thd_ignores_out_of_band_harmonics():
    n_samples = 8192
    fundamental_bin = 900
    hd2_dbc = -60.0
    signal = (
        _coherent_tone(n_samples, fundamental_bin)
        + _coherent_tone(n_samples, 2 * fundamental_bin, 10 ** (hd2_dbc / 20))
    )

    full_band = compute_spectrum(
        signal,
        fs=1.0,
        max_scale_range=[-1, 1],
        win_type="rectangular",
        side_bin=0,
        max_harmonic=5,
        nf_method=3,
        osr=1,
        assumed_sig_pwr_dbfs=0.0,
    )
    osr_four = compute_spectrum(
        signal,
        fs=1.0,
        max_scale_range=[-1, 1],
        win_type="rectangular",
        side_bin=0,
        max_harmonic=5,
        nf_method=3,
        osr=4,
        assumed_sig_pwr_dbfs=0.0,
    )

    assert full_band["metrics"]["harmonics_dbc"][0] == pytest.approx(hd2_dbc, abs=1e-9)
    assert full_band["metrics"]["thd_dbc"] == pytest.approx(hd2_dbc, abs=1e-9)
    assert osr_four["metrics"]["harmonics_dbc"][0] <= -149.0
    assert osr_four["metrics"]["thd_dbc"] <= -149.0
