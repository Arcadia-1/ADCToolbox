"""calibrate_weight_sine against known SAR weights and its own invariants.

Solver weights are in the unit-sine gauge: the fitted fundamental has
amplitude one, so an ADC with analog weights ``w`` driven by a sine of
amplitude ``A`` must return ``w / A``.
"""
import warnings

import numpy as np
import pytest

from adctoolbox.calibration import calibrate_weight_sine, calibrate_weight_sine_lite
from adctoolbox.models import sar_convert, sar_ideal_weights

N, BITS, AMPLITUDE = 8192, 12, 0.49
FREQ = 1021 / N


def _input(freq=FREQ, phase=0.4):
    return 0.5 + AMPLITUDE * np.sin(2 * np.pi * freq * np.arange(N) + phase)


@pytest.fixture(scope="module")
def binary_capture():
    weights = sar_ideal_weights(BITS)
    return sar_convert(_input(), weights), weights / AMPLITUDE


def _worst_error_lsb(weights, truth):
    return np.max(np.abs(weights - truth)) / truth[-1]


def test_binary_weights_are_recovered_in_the_unit_sine_gauge(binary_capture):
    bits, truth = binary_capture

    result = calibrate_weight_sine(bits, freq=FREQ)

    assert _worst_error_lsb(result["weight"], truth) < 0.05
    assert result["scale_convention"] == "solver_unit_sine"
    assert np.max(np.abs(result["ideal"][0])) == pytest.approx(1.0, rel=1e-3)


def test_sub_radix_two_weights_are_recovered():
    radix = 1.85 ** np.arange(BITS - 1, -1, -1)
    weights = radix / (radix.sum() + radix[-1])
    bits = sar_convert(_input(), weights)
    truth = weights / AMPLITUDE

    full = calibrate_weight_sine(bits, freq=FREQ, nominal_weights=weights)["weight"]
    lite = calibrate_weight_sine_lite(bits, FREQ)

    assert _worst_error_lsb(full, truth) < 0.3
    np.testing.assert_allclose(lite, full, rtol=1e-9)


def test_comparator_noise_degrades_the_weights_gracefully():
    weights = sar_ideal_weights(BITS)
    bits = sar_convert(_input(), weights, comparator_noise_rms=2e-4, rng=np.random.default_rng(3))

    result = calibrate_weight_sine(bits, freq=FREQ)

    # 2e-4 is 0.8 LSB rms of noise; averaging over 8192 samples leaves a
    # fraction of an LSB in the weights.
    assert _worst_error_lsb(result["weight"], weights / AMPLITUDE) < 0.6
    assert 60 < result["snr_db"] < 70


def test_noncoherent_tone_is_found_and_fitted():
    freq = 1021.37 / N
    weights = sar_ideal_weights(BITS)
    bits = sar_convert(_input(freq=freq), weights)

    result = calibrate_weight_sine(bits, freq=None)

    assert result["refined_frequency"] * N == pytest.approx(freq * N, abs=1e-4)
    assert _worst_error_lsb(result["weight"], weights / AMPLITUDE) < 0.05


def test_returned_waveforms_are_consistent_with_the_weights(binary_capture):
    bits, _ = binary_capture

    result = calibrate_weight_sine(bits, freq=FREQ)

    calibrated, ideal, error = result["calibrated_signal"][0], result["ideal"][0], result["error"][0]
    np.testing.assert_allclose(calibrated, bits @ result["weight"], atol=1e-12)
    np.testing.assert_allclose(error, calibrated - result["offset"] - ideal, atol=1e-12)
    assert result["snr_db"] == pytest.approx(
        10 * np.log10(np.mean(ideal**2) / np.mean(error**2)), abs=1e-9)
    assert result["enob"] == pytest.approx((result["snr_db"] - 1.76) / 6.02)
    # An ideal 12-bit quantizer leaves quantization noise only.
    assert result["snr_db"] == pytest.approx(6.02 * BITS + 1.76 + 20 * np.log10(AMPLITUDE / 0.5), abs=0.5)


def test_inverted_bit_column_returns_a_negated_weight(binary_capture):
    bits, truth = binary_capture
    inverted = bits.copy()
    inverted[:, 3] = 1 - inverted[:, 3]

    result = calibrate_weight_sine(inverted, freq=FREQ)

    assert result["weight"][3] == pytest.approx(-truth[3], rel=1e-4)
    others = np.delete(np.arange(BITS), 3)
    assert _worst_error_lsb(result["weight"][others], truth[others]) < 0.05


def test_plus_minus_one_coded_bits_return_half_weights(binary_capture):
    bits, truth = binary_capture

    result = calibrate_weight_sine(2.0 * bits - 1.0, freq=FREQ)

    assert _worst_error_lsb(2 * result["weight"], truth) < 0.05


def test_bit_column_order_does_not_change_the_fit(binary_capture):
    bits, _ = binary_capture
    order = np.random.default_rng(0).permutation(BITS)

    reference = calibrate_weight_sine(bits, freq=FREQ)["weight"]
    permuted = calibrate_weight_sine(bits[:, order], freq=FREQ)["weight"]

    np.testing.assert_allclose(permuted, reference[order], rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("amplitude", [0.49, 0.40, 0.30])
def test_weights_scale_inversely_with_the_training_amplitude(amplitude):
    weights = sar_ideal_weights(BITS)
    vin = 0.5 + amplitude * np.sin(2 * np.pi * FREQ * np.arange(N) + 0.4)

    result = calibrate_weight_sine(sar_convert(vin, weights), freq=FREQ)

    np.testing.assert_allclose(result["weight"] * amplitude, weights, atol=0.05 * weights[-1])


def test_small_swing_leaves_the_msb_pair_unidentifiable_and_says_so():
    # 0.5 +/- 0.15 never leaves [0.25, 0.75]: the second bit is always the
    # complement of the first. Their two weights cannot be separated, but
    # the reconstructed waveform must still fit.
    weights = sar_ideal_weights(BITS)
    vin = 0.5 + 0.15 * np.sin(2 * np.pi * FREQ * np.arange(N) + 0.4)
    bits = sar_convert(vin, weights)
    assert np.array_equal(bits[:, 1], 1 - bits[:, 0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = calibrate_weight_sine(bits, freq=FREQ)

    assert result["snr_db"] > 55
    # Only the difference of the two weights is observable.
    assert (result["weight"][0] - result["weight"][1]) * 0.15 == pytest.approx(
        weights[0] - weights[1], rel=1e-3)
    np.testing.assert_allclose(result["weight"][2:] * 0.15, weights[2:], atol=0.1 * weights[-1])
