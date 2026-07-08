"""Tests for MATLAB errsin compatibility wrappers."""

import numpy as np
import pytest
import matplotlib.pyplot as plt

from adctoolbox.aout import plot_error_hist_code, plot_error_hist_phase
from adctoolbox.fundamentals.fit_sine_4param import fit_sine_4param


def _test_signal(n=512, bin_index=17):
    freq = bin_index / n
    t = np.arange(n)
    signal = (
        0.42 * np.cos(2 * np.pi * freq * t + 0.3)
        + 0.002 * np.cos(2 * np.pi * 2 * freq * t - 0.4)
        + 0.1
    )
    return signal, freq


def test_plot_error_hist_phase_uses_matlab_errsin_contract():
    signal, freq = _test_signal()
    n_bins = 16

    emean, erms, phase_axis, anoi, pnoi, error, phase = plot_error_hist_phase(
        signal,
        bins=n_bins,
        freq=freq,
        disp=0,
    )

    fit = fit_sine_4param(signal, frequency_estimate=freq, max_iterations=0)

    assert phase_axis == pytest.approx(np.arange(n_bins) / n_bins * 360.0)
    assert np.nanmax(phase_axis) > 2 * np.pi
    assert error == pytest.approx(fit["fitted_signal"] - signal)
    assert phase.shape == signal.shape
    assert emean.shape == (n_bins,)
    assert erms.shape == (n_bins,)
    assert anoi >= 0.0
    assert pnoi >= 0.0


def test_plot_error_hist_code_returns_signal_value_axis():
    signal, freq = _test_signal()
    n_bins = 12

    emean, erms, code_axis, error, codes = plot_error_hist_code(
        signal,
        bins=n_bins,
        freq=freq,
        disp=0,
    )

    value_min = np.min(signal)
    value_max = np.max(signal)
    bin_width = (value_max - value_min) / n_bins
    expected_axis = value_min + (np.arange(n_bins) + 0.5) * bin_width

    assert code_axis == pytest.approx(expected_axis)
    assert not np.allclose(code_axis, np.arange(n_bins))
    assert codes == pytest.approx(signal)
    assert emean.shape == (n_bins,)
    assert erms.shape == (n_bins,)
    assert np.all(np.isfinite(error))


def test_plot_error_hist_phase_supports_erange_with_display():
    signal, freq = _test_signal()

    _emean, _erms, _phase_axis, _anoi, _pnoi, error, phase = plot_error_hist_phase(
        signal,
        bins=16,
        freq=freq,
        disp=1,
        erange=(0.0, 180.0),
    )

    assert error.shape == phase.shape
    assert len(error) < len(signal)
    assert np.all((phase >= 0.0) & (phase <= 180.0))
    plt.close("all")


def test_plot_error_hist_code_supports_erange_with_display():
    signal, freq = _test_signal()
    lo = float(np.percentile(signal, 25))
    hi = float(np.percentile(signal, 75))

    _emean, _erms, _code_axis, error, codes = plot_error_hist_code(
        signal,
        bins=12,
        freq=freq,
        disp=1,
        erange=(lo, hi),
    )

    assert error.shape == codes.shape
    assert len(error) < len(signal)
    assert np.all((codes >= lo) & (codes <= hi))
    plt.close("all")
