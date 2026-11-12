"""fit_sine_4param: converged default, least-squares A/B/C, and safe failure.

The default call used to run a single frequency-refinement step and return
A/B from that linearized step at the *previous* frequency. On a clean
non-coherent tone that left a residual as high as -29 dBc (N=256) and made
every error analysis built on the fit report the fit error as ADC error.
"""
import warnings

import numpy as np
import pytest

from adctoolbox import analyze_error_pdf, estimate_frequency, fit_sine_4param
from adctoolbox.fundamentals.fit_sine_4param import DEFAULT_MAX_ITERATIONS


def _tone(n, cycles, phase=0.7, amplitude=0.9, dc=0.1):
    t = np.arange(n)
    return amplitude * np.cos(2 * np.pi * cycles / n * t + phase) + dc


def _residual_dbc(result, amplitude=0.9):
    return 20 * np.log10(max(result["rmse"], 1e-300) / (amplitude / np.sqrt(2)))


def _three_param_rmse(signal, frequency):
    t = np.arange(len(signal))
    design = np.column_stack(
        (np.cos(2 * np.pi * frequency * t), np.sin(2 * np.pi * frequency * t), np.ones(len(signal)))
    )
    coefficients = np.linalg.lstsq(design, signal, rcond=None)[0]
    return np.sqrt(np.mean((signal - design @ coefficients) ** 2)), coefficients


@pytest.mark.parametrize(
    "n, cycles",
    [
        (256, 2.56),      # few cycles: the negative-frequency image biases the FFT start
        (256, 0.8),
        (1024, 12.636),
        (1024, 301.37),
        (4096, 1.3),
        (16, 3.3),
        (8192, 4000.21),  # close to Nyquist
    ],
)
def test_default_fit_converges_on_clean_noncoherent_tone(n, cycles):
    signal = _tone(n, cycles)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = fit_sine_4param(signal)

    assert result["converged"]
    assert 1 <= result["n_iterations"] < DEFAULT_MAX_ITERATIONS
    assert _residual_dbc(result) < -150
    assert result["frequency"] * n == pytest.approx(cycles, abs=1e-6)
    assert result["amplitude"] == pytest.approx(0.9, rel=1e-9)
    assert result["dc_offset"] == pytest.approx(0.1, abs=1e-9)


def test_returned_coefficients_are_least_squares_at_returned_frequency():
    # Stop after one step, before convergence: the returned A/B/C must still be
    # the 3-parameter optimum at the returned frequency.
    signal = _tone(256, 2.56)

    with pytest.warns(RuntimeWarning, match="did not converge"):
        result = fit_sine_4param(signal, max_iterations=1)

    best_rmse, coefficients = _three_param_rmse(signal, result["frequency"])
    assert result["rmse"] == pytest.approx(best_rmse, rel=1e-9)
    assert result["amplitude"] == pytest.approx(np.hypot(coefficients[0], coefficients[1]), rel=1e-9)
    assert result["dc_offset"] == pytest.approx(coefficients[2], abs=1e-12)
    np.testing.assert_allclose(result["residuals"], signal - result["fitted_signal"])


def test_fixed_frequency_fit_is_unchanged_three_parameter_solution():
    signal = _tone(512, 40.25)
    frequency = 40.0 / 512

    result = fit_sine_4param(signal, frequency_estimate=frequency, max_iterations=0)

    best_rmse, _ = _three_param_rmse(signal, frequency)
    assert result["frequency"] == frequency
    assert result["n_iterations"] == 0
    assert result["converged"]
    assert result["rmse"] == pytest.approx(best_rmse, rel=1e-12)


@pytest.mark.parametrize("kind", ["chirp", "noise"])
def test_refinement_never_returns_a_worse_fit_than_the_initial_estimate(kind):
    n = 1024
    t = np.arange(n)
    if kind == "chirp":
        signal = np.cos(2 * np.pi * (0.11 * t + 0.5 * (0.1 / n) * t * t))
    else:
        signal = np.random.default_rng(7).standard_normal(n)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_sine_4param(signal)

    initial_rmse, _ = _three_param_rmse(signal, result["initial_frequency"])
    assert result["rmse"] <= initial_rmse * (1 + 1e-9)
    assert 0 < result["frequency"] < 0.5


def test_diverging_refinement_falls_back_to_initial_frequency_and_warns():
    n = 1024
    t = np.arange(n)
    signal = np.cos(2 * np.pi * (0.11 * t + 0.5 * (0.1 / n) * t * t))

    with pytest.warns(RuntimeWarning, match="did not converge"):
        result = fit_sine_4param(signal)

    assert not result["converged"]


def test_top_bin_of_an_odd_length_record_is_found():
    # N = 4095: bin 2047 is an ordinary bin just below Nyquist (2047.5). The
    # FFT search used to stop one bin short and the fit then ran to the
    # frequency limit with an amplitude of several hundred thousand.
    n, cycles = 4095, 2047
    t = np.arange(n)
    signal = 0.5 * np.cos(2 * np.pi * cycles / n * t + 0.3)

    result = fit_sine_4param(signal)

    assert result["converged"]
    assert result["frequency"] * n == pytest.approx(cycles, abs=1e-6)
    assert result["amplitude"] == pytest.approx(0.5, rel=1e-9)


@pytest.mark.parametrize("n, cycles", [(4096, 0.5), (4095, 2046.9), (4096, 2047.9)])
def test_fit_at_the_edge_of_the_band_never_returns_a_runaway_amplitude(n, cycles):
    t = np.arange(n)
    signal = 0.5 * np.cos(2 * np.pi * cycles / n * t + 0.3)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fit_sine_4param(signal)

    assert 0 < result["frequency"] <= 0.5
    # Either a correct fit, or a flagged fallback with a bounded amplitude.
    assert result["amplitude"] < 2 * np.max(np.abs(signal))
    if not result["converged"]:
        assert any("did not converge" in str(w.message) for w in caught)
    else:
        assert result["amplitude"] == pytest.approx(0.5, rel=1e-6)


def test_error_pdf_of_low_frequency_tone_reports_quantization_noise_only():
    # 12.6 cycles of a 14-bit quantized tone. The only error is quantization,
    # sigma = 1/sqrt(12) LSB; the old default fit reported 1.43 LSB.
    n, bits = 1024, 14
    t = np.arange(n)
    ideal = 0.49 * np.cos(2 * np.pi * 0.01234 * t + 1.0) + 0.5
    signal = np.round(ideal * 2**bits) / 2**bits

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = analyze_error_pdf(signal, resolution=bits, full_scale=1.0, create_plot=False)

    assert result["sigma"] == pytest.approx(1 / np.sqrt(12), rel=0.05)


def test_estimate_frequency_takes_its_estimate_in_the_units_of_fs():
    fs, fin = 800e6, 100.37e6
    signal = np.sin(2 * np.pi * fin * np.arange(8192) / fs)

    assert estimate_frequency(signal, fs=fs) == pytest.approx(fin, rel=1e-9)
    assert estimate_frequency(signal, frequency_estimate=100.4e6, fs=fs) == pytest.approx(fin, rel=1e-9)
    assert estimate_frequency(signal, frequency_estimate=100.4e6 / fs) * fs == pytest.approx(fin, rel=1e-9)


@pytest.mark.parametrize("estimate", [0.0, -1e6, 500e6])
def test_estimate_frequency_rejects_an_estimate_outside_the_first_nyquist_zone(estimate):
    signal = np.sin(2 * np.pi * 0.1 * np.arange(256))

    with pytest.raises(ValueError, match="frequency_estimate"):
        estimate_frequency(signal, frequency_estimate=estimate, fs=800e6)
