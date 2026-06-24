"""Tests for fit controls exposed by AOUT residual analysis APIs."""

import warnings

import numpy as np
import pytest

from adctoolbox import (
    analyze_error_autocorr,
    analyze_error_by_phase,
    analyze_error_by_value,
    analyze_error_envelope_spectrum,
    analyze_error_pdf,
    analyze_error_spectrum,
)
from adctoolbox.aout.rearrange_error_by_phase import rearrange_error_by_phase
from adctoolbox.aout.rearrange_error_by_value import rearrange_error_by_value


FIT_KEYS = {
    "frequency",
    "amplitude",
    "phase",
    "dc_offset",
    "rmse",
    "converged",
    "n_iterations",
    "initial_frequency",
    "last_delta_freq",
}


def _signal_with_residual_harmonic(n=2048, freq_bin=37):
    t = np.arange(n)
    freq = freq_bin / n
    fundamental = 0.45 * np.sin(2 * np.pi * freq * t + 0.2) + 0.1
    harmonic = 1e-3 * np.sin(2 * np.pi * 2 * freq * t + 0.4)
    return fundamental + harmonic, freq


def _assert_fixed_frequency_fit(fit, freq):
    assert FIT_KEYS <= set(fit)
    assert fit["frequency"] == pytest.approx(freq)
    assert fit["initial_frequency"] == pytest.approx(freq)
    assert fit["converged"] is True
    assert fit["n_iterations"] == 0
    assert fit["last_delta_freq"] == pytest.approx(0.0)
    assert fit["rmse"] >= 0.0


def test_error_analysis_omits_fit_metadata_by_default():
    signal, freq = _signal_with_residual_harmonic()

    result = analyze_error_spectrum(
        signal,
        frequency=freq,
        create_plot=False,
        max_iterations=0,
    )

    assert "fit" not in result
    assert "error_signal" in result


def test_error_analysis_apis_return_optional_fit_metadata():
    signal, freq = _signal_with_residual_harmonic()
    cases = [
        (analyze_error_spectrum, {"frequency": freq, "create_plot": False}),
        (analyze_error_pdf, {"frequency": freq, "create_plot": False}),
        (analyze_error_autocorr, {"frequency": freq, "max_lag": 8, "create_plot": False}),
        (analyze_error_envelope_spectrum, {"frequency": freq, "create_plot": False}),
        (analyze_error_by_value, {"norm_freq": freq, "n_bins": 16, "create_plot": False}),
        (rearrange_error_by_value, {"norm_freq": freq, "n_bins": 16}),
        (analyze_error_by_phase, {"norm_freq": freq, "n_bins": 16, "create_plot": False}),
        (rearrange_error_by_phase, {"norm_freq": freq, "n_bins": 16}),
    ]

    for func, kwargs in cases:
        result = func(
            signal,
            **kwargs,
            max_iterations=0,
            tolerance=1e-9,
            return_fit=True,
        )

        assert "fit" in result, func.__name__
        _assert_fixed_frequency_fit(result["fit"], freq)


def test_analyze_error_spectrum_fit_options_reduce_near_dc_residual():
    n = 8192
    freq = 0.70 / n
    t = np.arange(n)
    signal = 0.5 * np.sin(2 * np.pi * freq * t) + 0.1

    with pytest.warns(RuntimeWarning, match="did not converge"):
        one_iter = analyze_error_spectrum(
            signal,
            create_plot=False,
            max_iterations=1,
            return_fit=True,
        )

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        five_iter = analyze_error_spectrum(
            signal,
            create_plot=False,
            max_iterations=5,
            return_fit=True,
        )

    runtime_warnings = [w for w in records if issubclass(w.category, RuntimeWarning)]
    assert runtime_warnings == []
    assert one_iter["fit"]["converged"] is False
    assert five_iter["fit"]["converged"] is True
    assert one_iter["fit"]["n_iterations"] == 1
    assert five_iter["fit"]["n_iterations"] > one_iter["fit"]["n_iterations"]
    assert five_iter["fit"]["frequency"] == pytest.approx(freq, abs=1e-12)
    assert np.std(five_iter["error_signal"]) < np.std(one_iter["error_signal"]) * 1e-8
