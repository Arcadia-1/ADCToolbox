"""calibrate_weight_sine_lite must not depend on the input's starting phase.

The solver fixed the cosine coefficient of the fundamental to one. A sine that
starts at zero phase has no cosine component, so the weights collapsed to
zero (sum 1e-5 instead of 2.04, 2048 LSB of error).
"""
import numpy as np
import pytest

from adctoolbox.calibration import calibrate_weight_sine, calibrate_weight_sine_lite
from adctoolbox.models.sar import sar_convert, sar_ideal_weights

N, BITS, FREQ = 8192, 12, 1021 / 8192
AMPLITUDE = 0.49


def _capture(phase, comparator_noise_rms=0.0):
    weights = sar_ideal_weights(BITS)
    vin = 0.5 + AMPLITUDE * np.sin(2 * np.pi * FREQ * np.arange(N) + phase)
    bits = sar_convert(
        vin, weights, comparator_noise_rms=comparator_noise_rms,
        rng=np.random.default_rng(3),
    )
    # Solver gauge: the fitted fundamental has unit amplitude.
    return bits, weights / AMPLITUDE


@pytest.mark.parametrize("phase", [0.0, 1e-6, np.pi / 2, np.pi, 0.5, -2.0])
def test_lite_recovers_weights_at_any_phase(phase):
    bits, truth = _capture(phase)

    weights = calibrate_weight_sine_lite(bits, FREQ)

    lsb = truth[-1]
    assert np.max(np.abs(weights - truth)) / lsb < 0.1
    assert np.sum(weights) == pytest.approx(np.sum(truth), rel=1e-4)


@pytest.mark.parametrize("phase", [0.0, np.pi])
def test_lite_matches_full_calibration_with_noise(phase):
    bits, truth = _capture(phase, comparator_noise_rms=2e-4)

    lite = calibrate_weight_sine_lite(bits, FREQ)
    full = calibrate_weight_sine(bits, freq=FREQ)["weight"]

    np.testing.assert_allclose(lite, full, rtol=0, atol=1e-6 * truth[0])


def test_lite_in_band_fit_recovers_weights_at_zero_phase():
    freq = 37 / N
    weights = sar_ideal_weights(BITS)
    vin = 0.5 + AMPLITUDE * np.sin(2 * np.pi * freq * np.arange(N))
    bits = sar_convert(vin, weights)

    fitted = calibrate_weight_sine_lite(bits, freq, osr=8)

    truth = weights / AMPLITUDE
    assert np.max(np.abs(fitted - truth)) / truth[-1] < 0.5
