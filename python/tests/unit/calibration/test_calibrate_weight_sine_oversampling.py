"""Independent finite-convolution checks for the oversampled fit objective."""

import numpy as np
import pytest
from scipy.signal import firwin, lfilter

from adctoolbox.calibration import calibrate_weight_sine, calibrate_weight_sine_lite
from adctoolbox.calibration import scale_calibration_output


def capture(n, freq, phase=0.3, dc=0.5, amplitude=0.35):
    x = dc + amplitude * np.sin(2 * np.pi * freq * np.arange(n) + phase)
    codes = np.floor(x * 256).astype(int)
    return (codes[:, None] >> np.arange(7, -1, -1)) & 1


def oracle_filter(values, osr):
    taps = firwin(16 * int(np.ceil(osr)) + 1, 0.5 / osr,
                  fs=1.0, window=("kaiser", 8.6))
    # Causal direct-form convolution, keeping only samples with a complete
    # input history. Independent of the implementation's FFT convolution.
    return lfilter(taps, [1.0], values, axis=0)[len(taps) - 1:]


@pytest.mark.parametrize("osr", [2, 4.5, 16])
def test_lite_matches_independent_filtered_design(osr):
    n, freq = 2048, 0.00713
    bits = capture(n, freq, phase=1.2)
    phase = 2 * np.pi * freq * np.arange(n)
    design = np.column_stack([bits, np.ones(n), np.sin(phase)])
    target = -np.cos(phase)
    coefficients = np.linalg.lstsq(oracle_filter(design, osr),
                                   oracle_filter(target, osr), rcond=None)[0]
    expected = coefficients[:8] / np.hypot(1, coefficients[-1])
    expected *= np.sign(expected.sum())

    actual = calibrate_weight_sine_lite(bits, freq, osr=osr)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("mode", ["known", "refined", "auto", "matlab_auto"])
def test_multi_capture_filtering_and_frequency_search(mode):
    lengths = np.array([4096, 3072])
    freqs = np.array([0.00513, 0.00827])
    amplitudes = [0.35, 0.22]
    bits = [capture(n, f, p, dc, a) for n, f, p, dc, a in
            zip(lengths, freqs, [0.3, 1.1], [0.50, 0.43], amplitudes)]
    options = {
        "known": dict(freq=freqs),
        "refined": dict(freq=freqs + 0.2 / lengths, force_search=True),
        "auto": dict(freq=None),
        "matlab_auto": dict(freq=None, frequency_policy="matlab"),
    }[mode]
    result = calibrate_weight_sine(bits, osr=8, harmonic_order=3, **options)
    np.testing.assert_allclose((result["refined_frequency"] - freqs) * lengths,
                               0, atol=0.003)
    np.testing.assert_array_equal(result["calibration_filter"]["valid_samples"],
                                  lengths - 128)
    for k, b in enumerate(bits):
        np.testing.assert_allclose(result["calibrated_signal"][k], b @ result["weight"])
        np.testing.assert_allclose(result["error"][k], b @ result["weight"]
                                   - result["offset"][k] - result["ideal"][k], atol=1e-14)
        assert len(result["ideal"][k]) == lengths[k]
        # A known input sine's amplitude/phase must remain in the original
        # time coordinates, despite the FIR's delay and valid-sample crop.
        phase = 2 * np.pi * freqs[k] * np.arange(lengths[k]) + [0.3, 1.1][k]
        expected = amplitudes[k] / amplitudes[0] * np.sin(phase)
        np.testing.assert_allclose(result["ideal"][k], expected, atol=0.002)
        ref = oracle_filter(result["ideal"][k], 8)
        err = oracle_filter(result["error"][k], 8)
        assert result["inband_snr_db"][k] == pytest.approx(
            10 * np.log10(np.mean(ref**2) / np.mean(err**2)), abs=1e-8)
    scaled = scale_calibration_output(result, scale=3)
    assert scaled["inband_snr_db"] == result["inband_snr_db"]


def test_full_fit_matches_independent_multirecord_weighted_least_squares():
    lengths, freqs = [1400, 1900], [0.0113, 0.0187]
    bits = [capture(n, f, p, dc) for n, f, p, dc in
            zip(lengths, freqs, [1.2, -0.4], [0.44, 0.53])]
    # One DC and two quadratures per capture, fixing only the first cosine.
    designs, targets = [], []
    for k, (n, f, b) in enumerate(zip(lengths, freqs, bits)):
        phase = 2 * np.pi * f * np.arange(n)
        design = np.zeros((n, 13))
        design[:, :8] = b
        design[:, 8 + k] = 1
        if k == 1:
            design[:, 10] = np.cos(phase)
        design[:, 11 + k] = np.sin(phase)
        designs.append(oracle_filter(design, 8))
        targets.append(oracle_filter(-np.cos(phase) if k == 0 else np.zeros(n), 8))
    coefficients = np.linalg.lstsq(np.vstack(designs), np.concatenate(targets), rcond=None)[0]
    expected = coefficients[:8] / np.hypot(1, coefficients[11])
    expected *= np.sign(expected.sum())
    result = calibrate_weight_sine(bits, freq=freqs, osr=8)
    # The cosine-dominant first reference selects the cosine-unity solve.
    np.testing.assert_allclose(result["weight"], expected, rtol=1e-9, atol=1e-11)


def test_osr_one_is_exactly_the_existing_path():
    bits = capture(1024, 0.0312)
    default = calibrate_weight_sine(bits, freq=0.0312)
    explicit = calibrate_weight_sine(bits, freq=0.0312, osr=1)
    assert set(default) == set(explicit)
    for key in ["weight", "offset", "calibrated_signal", "ideal", "error", "snr_db"]:
        np.testing.assert_array_equal(default[key], explicit[key])
    np.testing.assert_array_equal(calibrate_weight_sine_lite(bits, 0.0312),
                                  calibrate_weight_sine_lite(bits, 0.0312, osr=1))


@pytest.mark.parametrize("osr", [0, -1, 0.5, np.nan, np.inf])
@pytest.mark.parametrize("lite", [False, True])
def test_invalid_osr(osr, lite):
    fun = calibrate_weight_sine_lite if lite else calibrate_weight_sine
    with pytest.raises(ValueError, match="osr"):
        fun(capture(1024, 0.01), 0.01, osr=osr)


@pytest.mark.parametrize("freq", [-0.01, 0.5 / 16, 0.1, np.nan])
def test_training_tone_must_be_in_band(freq):
    with pytest.raises(ValueError, match="0 < freq"):
        calibrate_weight_sine(capture(1024, 0.01), freq=freq, force_search=False, osr=16)


def test_record_must_be_long_enough_after_filtering():
    with pytest.raises(ValueError, match="more than 257"):
        calibrate_weight_sine(capture(128, 0.01), freq=0.01, osr=16)
    with pytest.raises(ValueError, match="Too few samples"):
        calibrate_weight_sine(capture(260, 0.01), freq=0.01, osr=16)


def test_automatic_frequency_near_band_edge_stays_in_band():
    n, freq = 8192, 0.06249
    bits = capture(n, freq)
    result = calibrate_weight_sine(bits, freq=None, osr=8)
    assert 0 < result["refined_frequency"] < 0.5 / 8
    assert abs(result["refined_frequency"] - freq) * n < 0.003
    # The returned weights must be a static solution at the returned tone,
    # not coefficients from an unconstrained derivative correction.
    fixed = calibrate_weight_sine(bits, freq=result["refined_frequency"], osr=8)
    np.testing.assert_allclose(result["weight"], fixed["weight"], rtol=1e-10, atol=1e-12)
