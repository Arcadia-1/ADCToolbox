"""In-band weight calibration on a noise-shaping SAR, checked on held-out data.

The other oversampling tests compare the filtered fit with an independent
implementation of the same objective on a Nyquist quantizer. These check what
the feature is for: on an ADC whose quantization noise is shaped out of band,
the in-band fit must recover the analog weights where the full-band fit does
not, and that must show on a capture the fit never saw.
"""
import warnings

import numpy as np
import pytest

from adctoolbox.calibration import (
    calibrate_weight_sine,
    calibrate_weight_sine_lite,
    scale_calibration_output,
)
from adctoolbox.models import sar_convert, sar_ideal_weights
from adctoolbox.spectrum.compute_spectrum import compute_spectrum

N_SAMPLES, BURN_IN = 8192, 512
TRAIN_FREQ, TEST_FREQ = 23 / N_SAMPLES, 31 / N_SAMPLES


def noise_shaping_sar(inputs, analog_weights, order, comparator_noise_rms=0.0, seed=91):
    """Error-feedback loop of the given order around the SAR model.

    ``inputs`` has one column per independent capture. Returns the bit
    decisions, shape (samples, captures, bits), and the overload count.
    """
    rng = np.random.default_rng(seed)
    decisions = np.empty((*inputs.shape, len(analog_weights)), dtype=np.int8)
    error1 = np.zeros(inputs.shape[1])
    error2 = np.zeros_like(error1)
    overloads = 0
    for n, sample in enumerate(inputs):
        quantizer_input = sample - error1 if order == 1 else sample - 2 * error1 + error2
        overloads += np.count_nonzero((quantizer_input < 0) | (quantizer_input > 1))
        decisions[n] = sar_convert(
            quantizer_input, analog_weights, comparator_noise_rms=comparator_noise_rms, rng=rng,
        )
        quantized = decisions[n] @ analog_weights + analog_weights[-1] / 2
        error2, error1 = error1, quantized - quantizer_input
    return decisions, int(overloads)


def make_captures(bits, order, redundant_bit=None, comparator_noise_rms=0.0,
                  mismatch_seed=1, amplitudes=(0.30, 0.26), phases=(0.21, 1.3)):
    """Train and held-out captures of one mismatched noise-shaping SAR."""
    nominal = (sar_ideal_weights(bits, redundant_bit=redundant_bit)
               if redundant_bit else sar_ideal_weights(bits))
    mismatch = 1 + 0.04 * np.random.default_rng(mismatch_seed).standard_normal(len(nominal))
    analog = nominal * mismatch
    t = np.arange(N_SAMPLES + BURN_IN) - BURN_IN
    inputs = 0.5 + np.array(amplitudes) * np.sin(
        2 * np.pi * t[:, None] * np.array([TRAIN_FREQ, TEST_FREQ]) + np.array(phases))
    decisions, overloads = noise_shaping_sar(inputs, analog, order, comparator_noise_rms)
    assert overloads == 0, "test setup: the loop must not overload"
    return decisions[BURN_IN:, 0], decisions[BURN_IN:, 1], nominal, analog


def inband_sndr(test_bits, weights, osr):
    return compute_spectrum(
        test_bits @ weights, fs=1, osr=osr, win_type="hann", side_bin=1, max_scale_range=(0, 1),
    )["metrics"]["sndr_dbc"]


def fitted_weights(train_bits, nominal, osr, **kwargs):
    fit = calibrate_weight_sine(train_bits, freq=TRAIN_FREQ, nominal_weights=nominal, osr=osr, **kwargs)
    # Only the nominal scale is used, never the true analog weights.
    return scale_calibration_output(fit, target_weights=nominal)["weight"]


@pytest.mark.parametrize(
    "bits, order, osr, redundant_bit, comparator_noise_rms",
    [
        (6, 2, 16, None, 0.0),
        (6, 1, 16, None, 0.0),     # first-order shaping: less noise moved out of band
        (6, 2, 8, None, 0.0),
        (8, 2, 16, None, 0.0),
        (7, 2, 16, 3, 0.0),        # one redundant bit
        (6, 2, 16, None, 2e-3),    # comparator noise of about 1/8 LSB
        (8, 2, 16, None, 5e-4),
    ],
)
def test_inband_fit_matches_the_analog_weights_on_a_held_out_capture(
    bits, order, osr, redundant_bit, comparator_noise_rms,
):
    train, test, nominal, analog = make_captures(
        bits, order, redundant_bit=redundant_bit, comparator_noise_rms=comparator_noise_rms)
    analog_reference = analog * nominal.sum() / analog.sum()

    sndr = {
        "nominal": inband_sndr(test, nominal, osr),
        "full_band": inband_sndr(test, fitted_weights(train, nominal, 1), osr),
        "in_band": inband_sndr(test, fitted_weights(train, nominal, osr), osr),
        "analog": inband_sndr(test, analog_reference, osr),
    }

    # As good as knowing the analog weights, on data the fit never saw.
    assert sndr["in_band"] == pytest.approx(sndr["analog"], abs=1.5)
    # The full-band fit is pulled by the shaped out-of-band noise.
    assert sndr["in_band"] > sndr["full_band"] + 8
    assert sndr["in_band"] > sndr["nominal"] + 15


@pytest.mark.parametrize("bits, redundant_bit", [(6, None), (8, None), (7, 3)])
def test_inband_fit_recovers_the_weights_where_the_full_band_fit_does_not(bits, redundant_bit):
    train, _, nominal, analog = make_captures(bits, 2, redundant_bit=redundant_bit)
    analog_reference = analog * nominal.sum() / analog.sum()

    def worst_relative_error(weights):
        return np.max(np.abs(weights - analog_reference) / analog_reference)

    in_band = worst_relative_error(fitted_weights(train, nominal, 16))
    full_band = worst_relative_error(fitted_weights(train, nominal, 1))

    assert in_band < 0.007
    assert full_band > 5 * in_band


@pytest.mark.parametrize("mismatch_seed", [1, 2, 3])
def test_inband_fit_holds_for_different_mismatch_draws(mismatch_seed):
    train, test, nominal, analog = make_captures(6, 2, mismatch_seed=mismatch_seed)
    analog_reference = analog * nominal.sum() / analog.sum()

    in_band = inband_sndr(test, fitted_weights(train, nominal, 16), 16)

    assert in_band == pytest.approx(inband_sndr(test, analog_reference, 16), abs=1.5)
    assert in_band > inband_sndr(test, fitted_weights(train, nominal, 1), 16) + 8


@pytest.mark.parametrize("frequency_policy", ["python", "matlab"])
def test_automatic_frequency_finds_the_tone_under_shaped_noise(frequency_policy):
    # Out of band the shaped noise carries far more power than the tone; the
    # coarse search must still land on the in-band tone.
    train, _, nominal, _ = make_captures(6, 2)

    known = calibrate_weight_sine(train, freq=TRAIN_FREQ, nominal_weights=nominal, osr=16)
    auto = calibrate_weight_sine(
        train, freq=None, nominal_weights=nominal, osr=16, frequency_policy=frequency_policy)

    assert auto["refined_frequency"] * N_SAMPLES == pytest.approx(TRAIN_FREQ * N_SAMPLES, abs=1e-3)
    np.testing.assert_allclose(auto["weight"], known["weight"], rtol=1e-4)


@pytest.mark.parametrize("frequency_policy", ["python", "matlab"])
@pytest.mark.parametrize("tone_bin", [513, 900, 2000])
def test_automatic_frequency_rejects_a_tone_outside_the_band(frequency_policy, tone_bin):
    # osr = 16 keeps bins below 256. The search used to return whatever was
    # left in band: weights hundreds of times too large, without a warning.
    weights = sar_ideal_weights(8)
    vin = 0.5 + 0.45 * np.sin(2 * np.pi * tone_bin / N_SAMPLES * np.arange(N_SAMPLES) + 0.3)
    bits = sar_convert(vin, weights)

    with pytest.raises(ValueError, match="outside the calibration band"):
        calibrate_weight_sine(bits, freq=None, osr=16, frequency_policy=frequency_policy)


def test_waveform_fields_keep_the_capture_length_and_report_the_filter():
    train, _, nominal, _ = make_captures(6, 2)
    osr = 16

    result = calibrate_weight_sine(train, freq=TRAIN_FREQ, nominal_weights=nominal, osr=osr)

    for field in ("calibrated_signal", "ideal", "error"):
        assert len(result[field]) == 1
        assert result[field][0].shape == (N_SAMPLES,), field
    np.testing.assert_allclose(result["calibrated_signal"][0], train @ result["weight"])
    numtaps = 16 * osr + 1
    assert result["calibration_filter"]["kind"] == "fir_lowpass"
    assert result["calibration_filter"]["osr"] == osr
    assert result["calibration_filter"]["cutoff_normalized"] == 0.5 / osr
    assert result["calibration_filter"]["numtaps"] == numtaps
    np.testing.assert_array_equal(
        result["calibration_filter"]["valid_samples"], [N_SAMPLES - numtaps + 1])


def test_inband_residual_metric_excludes_the_shaped_noise():
    train, _, nominal, _ = make_captures(6, 2)

    result = calibrate_weight_sine(train, freq=TRAIN_FREQ, nominal_weights=nominal, osr=16)

    # Full band the residual is dominated by shaped quantization noise.
    assert result["snr_db"] < 35
    assert result["inband_snr_db"] > result["snr_db"] + 40
    assert result["inband_enob"] == pytest.approx((result["inband_snr_db"] - 1.76) / 6.02)


@pytest.mark.parametrize("phase", [0.21, 0.0, np.pi / 2])
def test_lite_and_full_inband_fits_agree_at_any_phase(phase):
    train, _, nominal, _ = make_captures(6, 2, phases=(phase, 1.3))

    full = calibrate_weight_sine(train, freq=TRAIN_FREQ, nominal_weights=nominal, osr=16)["weight"]
    lite = calibrate_weight_sine_lite(train, TRAIN_FREQ, osr=16)

    np.testing.assert_allclose(lite, full, rtol=1e-9)


def test_bit_column_order_does_not_change_the_fit():
    train, _, nominal, _ = make_captures(6, 2)
    order = np.array([3, 0, 5, 1, 4, 2])

    reference = calibrate_weight_sine(train, freq=TRAIN_FREQ, nominal_weights=nominal, osr=16)["weight"]
    permuted = calibrate_weight_sine(
        train[:, order], freq=TRAIN_FREQ, nominal_weights=nominal[order], osr=16)["weight"]

    np.testing.assert_allclose(permuted, reference[order], rtol=1e-10)


def test_unit_sine_gauge_scales_with_the_training_amplitude():
    # Solver weights are w / A. Two captures of the same ADC at different
    # amplitudes must agree once multiplied by their amplitude.
    scaled = []
    for amplitude in (0.30, 0.26):
        train, _, nominal, analog = make_captures(6, 2, amplitudes=(amplitude, 0.26))
        fit = calibrate_weight_sine(train, freq=TRAIN_FREQ, nominal_weights=nominal, osr=16)
        scaled.append(fit["weight"] * amplitude)

    np.testing.assert_allclose(scaled[0], analog, rtol=0.01)
    np.testing.assert_allclose(scaled[1], scaled[0], rtol=0.01)


def test_harmonic_columns_do_not_disturb_a_clean_inband_fit():
    train, test, nominal, _ = make_captures(6, 2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = fitted_weights(train, nominal, 16)
        third = fitted_weights(train, nominal, 16, harmonic_order=3)

    assert inband_sndr(test, third, 16) == pytest.approx(inband_sndr(test, first, 16), abs=1.0)


def test_two_captures_share_weights_and_placeholder_frequency_is_estimated():
    # Both captures of the same ADC are fitted jointly. The second frequency
    # is left as a 0 placeholder and must be found inside the band.
    train, test, nominal, analog = make_captures(6, 2)
    analog_reference = analog * nominal.sum() / analog.sum()

    known = calibrate_weight_sine(
        [train, test], freq=[TRAIN_FREQ, TEST_FREQ], nominal_weights=nominal, osr=16)
    mixed = calibrate_weight_sine(
        [train, test], freq=[TRAIN_FREQ, 0], nominal_weights=nominal, osr=16)

    np.testing.assert_allclose(
        mixed["refined_frequency"], [TRAIN_FREQ, TEST_FREQ], atol=1e-3 / N_SAMPLES)
    np.testing.assert_allclose(mixed["weight"], known["weight"], rtol=1e-4)
    weights = scale_calibration_output(known, target_weights=nominal)["weight"]
    assert np.max(np.abs(weights - analog_reference) / analog_reference) < 0.007
    assert len(known["inband_snr_db"]) == 2
    assert min(known["inband_snr_db"]) > 60


@pytest.mark.parametrize("osr, margin_db", [(4.5, 5), (12.3, 8)])
def test_non_integer_osr_is_accepted_and_still_beats_the_full_band_fit(osr, margin_db):
    # A wider band keeps more of the shaped noise, so the advantage over the
    # full-band fit shrinks with the OSR; the analog reference is still met.
    train, test, nominal, analog = make_captures(6, 2)
    analog_reference = analog * nominal.sum() / analog.sum()

    in_band = inband_sndr(test, fitted_weights(train, nominal, osr), osr)

    assert in_band == pytest.approx(inband_sndr(test, analog_reference, osr), abs=1.5)
    assert in_band > inband_sndr(test, fitted_weights(train, nominal, 1), osr) + margin_db
