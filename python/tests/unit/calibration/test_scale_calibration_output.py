import numpy as np
import pytest

from adctoolbox import analyze_spectrum, calibrate_weight_sine, scale_calibration_output
from adctoolbox.models import (
    sar_apply_cap_mismatch,
    sar_convert,
    sar_ideal_weights,
    sar_reconstruct,
)


def _calibration_case():
    n_samples = 2048
    train_bin = 499
    test_bin = 619
    sine_peak = 0.49
    n = np.arange(n_samples)

    nominal_weights = sar_ideal_weights(12)
    actual_weights = sar_apply_cap_mismatch(
        nominal_weights, sigma=0.01, rng=np.random.default_rng(3)
    )

    vin_train = 0.5 + sine_peak * np.sin(2 * np.pi * train_bin * n / n_samples)
    bits_train = sar_convert(vin_train, actual_weights, quant_range=(0, 1))
    result = calibrate_weight_sine(
        bits_train,
        freq=train_bin / n_samples,
        nominal_weights=nominal_weights,
    )

    vin_test = 0.5 + sine_peak * np.sin(2 * np.pi * test_bin * n / n_samples)
    bits_test = sar_convert(vin_test, actual_weights, quant_range=(0, 1))
    before = sar_reconstruct(bits_test, nominal_weights, quant_range=(0, 1))
    raw_after = bits_test.astype(float) @ result["weight"]
    return nominal_weights, result, before, raw_after, bits_test, sine_peak


def test_calibration_result_declares_solver_unit_sine_scale():
    _, result, *_ = _calibration_case()

    assert result["scale_convention"] == "solver_unit_sine"


def test_scale_calibration_output_target_weights_scales_waveform_fields():
    nominal_weights, result, *_ = _calibration_case()

    scaled = scale_calibration_output(result, target_weights=nominal_weights)
    scale = scaled["scale_factor"]

    assert scaled["scale_convention"] == "adc_reference_scale"
    assert scaled["source_scale_convention"] == "solver_unit_sine"
    np.testing.assert_allclose(
        np.sum(scaled["weight"]),
        np.sum(nominal_weights),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(scaled["weight"], result["weight"] * scale)
    np.testing.assert_allclose(scaled["offset"], result["offset"] * scale)
    np.testing.assert_allclose(
        scaled["calibrated_signal"][0],
        result["calibrated_signal"][0] * scale,
    )
    np.testing.assert_allclose(scaled["ideal"][0], result["ideal"][0] * scale)
    np.testing.assert_allclose(scaled["error"][0], result["error"][0] * scale)

    assert scaled["snr_db"] == result["snr_db"]
    assert scaled["enob"] == result["enob"]
    assert scaled["refined_frequency"] == result["refined_frequency"]


def test_scale_calibration_output_target_sine_peak_matches_direct_scale():
    nominal_weights, result, *_rest, sine_peak = _calibration_case()

    by_peak = scale_calibration_output(result, target_sine_peak=sine_peak)
    by_weights = scale_calibration_output(result, target_weights=nominal_weights)

    assert by_peak["scale_factor"] == pytest.approx(sine_peak)
    assert by_weights["scale_factor"] == pytest.approx(sine_peak, abs=1e-3)


def test_scaled_calibration_restores_dbfs_nsd_reference_without_changing_ratios():
    nominal_weights, result, before, raw_after, bits_test, _ = _calibration_case()

    scaled = scale_calibration_output(result, target_weights=nominal_weights)
    scaled_after = bits_test.astype(float) @ scaled["weight"]

    common = dict(
        fs=1.0,
        osr=1,
        max_scale_range=(0.0, 1.0),
        win_type="rectangular",
        side_bin=0,
        max_harmonic=5,
        nf_method=3,
        create_plot=False,
    )
    before_metrics = analyze_spectrum(before, **common)
    raw_metrics = analyze_spectrum(raw_after, **common)
    scaled_metrics = analyze_spectrum(scaled_after, **common)

    for key in ("sndr_dbc", "snr_dbc", "sfdr_dbc", "thd_dbc", "enob"):
        assert scaled_metrics[key] == pytest.approx(raw_metrics[key], abs=1e-9)

    expected_db_shift = 20 * np.log10(scaled["scale_factor"])
    for key in ("sig_pwr_dbfs", "noise_floor_dbfs", "nsd_dbfs_hz"):
        assert scaled_metrics[key] - raw_metrics[key] == pytest.approx(
            expected_db_shift, abs=1e-6
        )

    assert scaled_metrics["sndr_dbc"] > before_metrics["sndr_dbc"] + 5.0
    assert scaled_metrics["nsd_dbfs_hz"] < before_metrics["nsd_dbfs_hz"] - 2.0
    assert raw_metrics["nsd_dbfs_hz"] > before_metrics["nsd_dbfs_hz"]


def test_scale_calibration_output_rejects_ambiguous_or_invalid_inputs():
    _, result, *_ = _calibration_case()

    with pytest.raises(ValueError, match="exactly one"):
        scale_calibration_output(result)
    with pytest.raises(ValueError, match="exactly one"):
        scale_calibration_output(result, scale=1.0, target_sine_peak=0.5)
    with pytest.raises(ValueError, match="finite and non-zero"):
        scale_calibration_output(result, scale=0.0)
    with pytest.raises(ValueError, match="target_weights must not be empty"):
        scale_calibration_output(result, target_weights=[])


def test_target_weights_scale_ignores_bits_the_capture_never_exercised():
    # The input stays below mid-scale, so the MSB never toggles and its solved
    # weight is forced to zero. Summing the target over all bits but the
    # solution over the observed ones doubled the scale (0.40 instead of 0.20).
    n_samples, n_bits, freq = 2048, 10, 255 / 2048
    weights = sar_ideal_weights(n_bits)
    vin = 0.25 + 0.2 * np.sin(2 * np.pi * freq * np.arange(n_samples) + 0.3)
    bits = sar_convert(vin, weights)

    result = calibrate_weight_sine(bits, freq=freq)
    scaled = scale_calibration_output(result, target_weights=weights)

    assert list(result["rank_patch"]["unmapped_bits"]) == [0]
    assert scaled["scale_factor"] == pytest.approx(0.2, rel=1e-3)
    np.testing.assert_allclose(scaled["weight"][1:], weights[1:], rtol=0, atol=0.05 * weights[-1])
    assert np.max(np.abs(scaled["ideal"][0])) == pytest.approx(0.2, rel=1e-3)
