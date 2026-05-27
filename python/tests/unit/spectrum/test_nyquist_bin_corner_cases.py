"""Regression tests for Nyquist-bin spectrum corner cases."""

import numpy as np
import pytest

from adctoolbox.spectrum._bin_ranges import rfft_inband_bin_count
from adctoolbox.spectrum.compute_spectrum import compute_spectrum


_TONE_AMPLITUDE = 0.4
_MAX_SCALE_RANGE = [-0.5, 0.5]


def _ordinary_bin_cases():
    """All integer sine bins currently inside the default in-band search."""
    cases = []
    for n_fft in range(4, 33):
        # Boundary bins are covered by dedicated cases below.
        for fundamental_bin in range(1, n_fft // 2):
            cases.append((n_fft, fundamental_bin))
    return cases


def _odd_highest_positive_bin_cases():
    return [(n_fft, n_fft // 2) for n_fft in range(5, 33, 2)]


def _even_nyquist_bin_cases():
    return [(n_fft, n_fft // 2) for n_fft in range(4, 33, 2)]


def _small_nyquist_harmonic_cases():
    """All N=4..32 cases where HD2..HD5 lands exactly on Nyquist."""
    cases = []
    for n_fft in range(4, 33):
        if n_fft % 2:
            continue
        nyquist_bin = n_fft // 2
        for harmonic_order in range(2, 6):
            if nyquist_bin % harmonic_order == 0:
                fundamental_bin = nyquist_bin // harmonic_order
                if fundamental_bin >= 1:
                    cases.append((n_fft, fundamental_bin, harmonic_order))
    return cases


def _expected_sine_sig_pwr_dbfs(amplitude=_TONE_AMPLITUDE):
    return 20 * np.log10(amplitude / 0.5)


def _expected_nyquist_sig_pwr_dbfs(amplitude=_TONE_AMPLITUDE):
    """Nyquist cosine has twice the power of an equal-peak sine."""
    return _expected_sine_sig_pwr_dbfs(amplitude) + 10 * np.log10(2.0)


def _compute(signal, max_harmonic=5):
    return compute_spectrum(
        signal,
        fs=1.0,
        max_scale_range=_MAX_SCALE_RANGE,
        win_type="rectangular",
        max_harmonic=max_harmonic,
        side_bin=0,
    )


def _tone_with_nyquist_harmonic(
    n_fft,
    fundamental_bin,
    harmonic_order,
    harmonic_peak_amp_db,
):
    """Build a coherent tone whose selected harmonic lands on Nyquist.

    ``harmonic_peak_amp_db`` is a peak-amplitude ratio relative to the
    fundamental. When the harmonic lands on Nyquist, its power dBc is 3.0103 dB
    higher than that peak-amplitude ratio.
    """
    assert harmonic_order * fundamental_bin == n_fft // 2
    n = np.arange(n_fft)
    a_fund = _TONE_AMPLITUDE
    a_harm = a_fund * 10 ** (harmonic_peak_amp_db / 20)
    signal = (
        a_fund * np.sin(2 * np.pi * fundamental_bin * n / n_fft)
        + a_harm * np.cos(np.pi * n)
    )
    return signal


def _expected_nyquist_harmonic_dbc(harmonic_peak_amp_db):
    """Convert a peak-amplitude ratio to power dBc for a Nyquist harmonic."""
    return harmonic_peak_amp_db + 10 * np.log10(2.0)


@pytest.mark.parametrize(
    "n_fft,expected_count",
    [
        (4, 3),    # bins 0..2, including Nyquist
        (5, 3),    # bins 0..2, no Nyquist
        (15, 8),   # bins 0..7, highest positive bin included
        (16, 9),   # bins 0..8, including Nyquist
    ],
)
def test_rfft_inband_bin_count_includes_valid_upper_edge(n_fft, expected_count):
    assert rfft_inband_bin_count(n_fft, osr=1) == expected_count


@pytest.mark.parametrize(
    "n_fft,osr,expected_count",
    [
        (16, 2, 5),   # bins 0..4
        (16, 3, 3),   # bins 0..2
        (17, 2, 5),   # bins 0..4
        (17, 3, 3),   # bins 0..2
        (32, 4, 5),   # bins 0..4
        (33, 4, 5),   # bins 0..4
    ],
)
def test_rfft_inband_bin_count_includes_osr_band_edge(n_fft, osr, expected_count):
    assert rfft_inband_bin_count(n_fft, osr=osr) == expected_count


@pytest.mark.parametrize("n_fft,fundamental_bin", _ordinary_bin_cases())
def test_integer_sine_bins_from_n4_to_n32_are_located_and_scaled(
    n_fft,
    fundamental_bin,
):
    n = np.arange(n_fft)
    signal = _TONE_AMPLITUDE * np.sin(2 * np.pi * fundamental_bin * n / n_fft)

    result = _compute(signal)

    assert result["plot_data"]["fundamental_bin"] == fundamental_bin
    assert result["metrics"]["sig_pwr_dbfs"] == pytest.approx(
        _expected_sine_sig_pwr_dbfs(),
        abs=1e-9,
    )


@pytest.mark.parametrize(
    "n_fft,osr,band_edge_bin",
    [
        (16, 2, 4),
        (16, 3, 2),
        (17, 2, 4),
        (17, 3, 2),
        (32, 4, 4),
        (33, 4, 4),
    ],
)
def test_osr_gt_one_band_edge_tone_is_included(n_fft, osr, band_edge_bin):
    n = np.arange(n_fft)
    signal = _TONE_AMPLITUDE * np.sin(2 * np.pi * band_edge_bin * n / n_fft)

    result = compute_spectrum(
        signal,
        fs=1.0,
        osr=osr,
        max_scale_range=_MAX_SCALE_RANGE,
        win_type="rectangular",
        side_bin=0,
    )

    assert result["plot_data"]["fundamental_bin"] == band_edge_bin
    assert result["metrics"]["sig_pwr_dbfs"] == pytest.approx(
        _expected_sine_sig_pwr_dbfs(),
        abs=1e-9,
    )


@pytest.mark.parametrize("n_fft,fundamental_bin", _odd_highest_positive_bin_cases())
def test_odd_n_highest_positive_bin_from_n4_to_n32_is_in_band(
    n_fft,
    fundamental_bin,
):
    n = np.arange(n_fft)
    signal = _TONE_AMPLITUDE * np.sin(2 * np.pi * fundamental_bin * n / n_fft)

    result = _compute(signal)

    assert result["plot_data"]["fundamental_bin"] == fundamental_bin
    assert result["metrics"]["sig_pwr_dbfs"] == pytest.approx(
        _expected_sine_sig_pwr_dbfs(),
        abs=1e-9,
    )


@pytest.mark.parametrize(
    "n_fft,fundamental_bin,expected_sig_pwr_dbfs",
    [
        (16, 8, _expected_nyquist_sig_pwr_dbfs()),
        (17, 8, _expected_sine_sig_pwr_dbfs()),
        (32, 16, _expected_nyquist_sig_pwr_dbfs()),
        (33, 16, _expected_sine_sig_pwr_dbfs()),
    ],
)
def test_coherent_averaging_reports_boundary_tones(
    n_fft,
    fundamental_bin,
    expected_sig_pwr_dbfs,
):
    n = np.arange(n_fft)
    if fundamental_bin == n_fft // 2 and n_fft % 2 == 0:
        signal = _TONE_AMPLITUDE * np.cos(np.pi * n)
    else:
        signal = _TONE_AMPLITUDE * np.sin(2 * np.pi * fundamental_bin * n / n_fft)
    data = np.vstack([signal, signal])

    result = compute_spectrum(
        data,
        fs=1.0,
        max_scale_range=_MAX_SCALE_RANGE,
        win_type="rectangular",
        side_bin=0,
        coherent_averaging=True,
    )

    assert result["plot_data"]["fundamental_bin"] == fundamental_bin
    assert result["metrics"]["sig_pwr_dbfs"] == pytest.approx(
        expected_sig_pwr_dbfs,
        abs=1e-9,
    )


@pytest.mark.parametrize("n_fft,nyquist_bin", _even_nyquist_bin_cases())
def test_even_n_nyquist_input_from_n4_to_n32_is_reported_as_boundary_tone(
    n_fft,
    nyquist_bin,
):
    n = np.arange(n_fft)
    signal = _TONE_AMPLITUDE * np.cos(np.pi * n)

    result = _compute(signal)

    assert result["plot_data"]["fundamental_bin"] == nyquist_bin
    assert result["metrics"]["sig_pwr_dbfs"] == pytest.approx(
        _expected_nyquist_sig_pwr_dbfs(),
        abs=1e-9,
    )


@pytest.mark.parametrize(
    "n_fft,fundamental_bin,harmonic_order",
    _small_nyquist_harmonic_cases(),
)
@pytest.mark.parametrize("harmonic_peak_amp_db", [-60.0, -20.0])
def test_n4_to_n32_harmonics_landing_on_nyquist_have_boundary_power(
    n_fft,
    fundamental_bin,
    harmonic_order,
    harmonic_peak_amp_db,
):
    signal = _tone_with_nyquist_harmonic(
        n_fft,
        fundamental_bin,
        harmonic_order,
        harmonic_peak_amp_db,
    )

    result = _compute(signal)

    assert result["plot_data"]["fundamental_bin"] == fundamental_bin
    assert result["plot_data"]["harmonic_bins"][harmonic_order - 2] == n_fft // 2
    assert result["metrics"]["harmonics_dbc"][harmonic_order - 2] == pytest.approx(
        _expected_nyquist_harmonic_dbc(harmonic_peak_amp_db),
        abs=1e-9,
    )


@pytest.mark.parametrize(
    "n_fft,fundamental_bin,harmonic_order",
    [
        (1024, 256, 2),   # Fin = Fs/4 -> HD2 at Fs/2
        (1536, 256, 3),   # Fin = Fs/6 -> HD3 at Fs/2
        (2048, 256, 4),   # Fin = Fs/8 -> HD4 at Fs/2
    ],
)
def test_nyquist_harmonic_bin_and_power_are_theoretical(
    n_fft,
    fundamental_bin,
    harmonic_order,
):
    harmonic_peak_amp_db = -60.0
    signal = _tone_with_nyquist_harmonic(
        n_fft,
        fundamental_bin,
        harmonic_order,
        harmonic_peak_amp_db,
    )

    result = _compute(signal)

    assert result["plot_data"]["fundamental_bin"] == fundamental_bin
    assert result["plot_data"]["harmonic_bins"][harmonic_order - 2] == n_fft // 2
    assert result["metrics"]["harmonics_dbc"][harmonic_order - 2] == pytest.approx(
        _expected_nyquist_harmonic_dbc(harmonic_peak_amp_db),
        abs=1e-9,
    )


def test_interior_harmonic_does_not_have_nyquist_boundary_gain():
    n_fft = 1024
    n = np.arange(n_fft)
    fundamental_bin = 128
    harmonic_bin = 3 * fundamental_bin
    harmonic_peak_amp_db = -60.0
    a_fund = _TONE_AMPLITUDE
    a_harm = a_fund * 10 ** (harmonic_peak_amp_db / 20)
    signal = (
        a_fund * np.sin(2 * np.pi * fundamental_bin * n / n_fft)
        + a_harm * np.sin(2 * np.pi * harmonic_bin * n / n_fft)
    )

    result = _compute(signal)

    assert result["plot_data"]["harmonic_bins"][1] == harmonic_bin
    assert result["metrics"]["harmonics_dbc"][1] == pytest.approx(
        harmonic_peak_amp_db,
        abs=1e-9,
    )


def test_sndr_includes_nyquist_bin_distortion_when_harmonic_is_included():
    harmonic_peak_amp_db = -60.0
    signal = _tone_with_nyquist_harmonic(
        n_fft=1024,
        fundamental_bin=256,
        harmonic_order=2,
        harmonic_peak_amp_db=harmonic_peak_amp_db,
    )

    result = _compute(signal)

    expected_distortion_dbc = _expected_nyquist_harmonic_dbc(harmonic_peak_amp_db)
    assert result["metrics"]["sndr_dbc"] == pytest.approx(
        -expected_distortion_dbc,
        abs=1e-9,
    )
