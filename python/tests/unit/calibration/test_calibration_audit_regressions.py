"""Regression tests for calibration solver, rank-patching, and dither HDC fixes."""

import warnings
import numpy as np
import pytest

from adctoolbox.calibration import (
    DitherHDCTracker,
    calibrate_weight_sine,
    calibrate_weight_sine_lite,
    diagnose_calibration_matrix,
    scale_calibration_output,
)
from adctoolbox.models.sar import sar_convert, sar_ideal_weights


@pytest.mark.parametrize("phase", [-np.pi / 4, 3 * np.pi / 4, 0.0, np.pi / 2])
def test_frequency_search_converges_across_basis_switch_phases(phase):
    """Frequency search must not invert step sign when switching cos/sin gauge."""
    n_samples = 2048
    n = np.arange(n_samples)
    true_freq = 101.3 / n_samples
    vin = 0.5 + 0.49 * np.sin(2 * np.pi * true_freq * n + phase)
    bits = sar_convert(vin, sar_ideal_weights(12))

    res = calibrate_weight_sine(
        bits, freq=true_freq - 0.2 / n_samples, force_search=True
    )
    assert abs(res["refined_frequency"] - true_freq) * n_samples < 1e-4
    assert res["snr_db"] > 72.0


def test_frequency_search_basis_switch_oversampled_and_multi_dataset():
    """Basis-flip gauge scaling also holds for multi-dataset and osr > 1."""
    n_samples = 2048
    n = np.arange(n_samples)
    w12 = sar_ideal_weights(12)

    f1, f2 = 101.3 / n_samples, 173.4 / n_samples
    b1 = sar_convert(0.5 + 0.49 * np.sin(2 * np.pi * f1 * n - np.pi / 4), w12)
    b2 = sar_convert(0.5 + 0.49 * np.sin(2 * np.pi * f2 * n + 0.6), w12)
    res_md = calibrate_weight_sine(
        [b1, b2],
        freq=[f1 - 0.2 / n_samples, f2 + 0.2 / n_samples],
        force_search=True,
    )
    assert abs(res_md["refined_frequency"][0] - f1) * n_samples < 1e-4
    assert abs(res_md["refined_frequency"][1] - f2) * n_samples < 1e-4
    assert min(res_md["snr_db"]) > 72.0

    f_osr = 19.3 / n_samples
    b_osr = sar_convert(0.5 + 0.49 * np.sin(2 * np.pi * f_osr * n - np.pi / 4), w12)
    res_osr = calibrate_weight_sine(
        b_osr, freq=f_osr - 0.2 / n_samples, force_search=True, osr=8
    )
    assert abs(res_osr["refined_frequency"] - f_osr) * n_samples < 1e-3
    assert res_osr["inband_snr_db"] > 75.0


def test_multi_dataset_piecewise_constant_bit_is_detected_and_patched():
    """A bit constant within each dataset lies in the per-dataset DC span."""
    n_samples = 2048
    n = np.arange(n_samples)
    f1, f2 = 101.0 / n_samples, 137.0 / n_samples
    w10 = sar_ideal_weights(10)
    b1 = sar_convert(0.25 + 0.20 * np.sin(2 * np.pi * f1 * n + 0.3), w10)
    b2 = sar_convert(0.75 + 0.20 * np.sin(2 * np.pi * f2 * n + 1.1), w10)

    diag = diagnose_calibration_matrix([b1, b2])
    assert diag["rank_with_offset"] < 10 + 2

    with pytest.warns(UserWarning, match="constant or otherwise unobservable"):
        res = calibrate_weight_sine([b1, b2], freq=[f1, f2])

    assert res["rank_patch"]["applied"] is True
    assert 0 in res["rank_patch"]["unmapped_bits"]
    assert res["weight"][0] == 0.0
    assert np.all(np.abs(res["weight"]) < 10.0)
    assert np.all(np.abs(res["offset"]) < 10.0)


def test_complementary_equal_weight_bits_do_not_collapse_rank():
    """Merging b1 = 1 - b0 with equal nominal weight must not annihilate the column."""
    n_samples = 2048
    n = np.arange(n_samples)
    freq = 101.0 / n_samples
    nom = np.array([256, 256, 128, 64, 32, 16, 8, 4, 2, 1], dtype=float)
    codes9 = np.floor((0.5 + 0.45 * np.sin(2 * np.pi * freq * n + 0.3)) * 512).astype(int)
    b0 = ((codes9 >> 8) & 1).astype(float)
    b1 = 1.0 - b0
    tail = np.column_stack([((codes9 >> k) & 1).astype(float) for k in range(7, -1, -1)])
    bits = np.column_stack([b0, b1, tail])

    with pytest.warns(UserWarning, match="unobservable"):
        res = calibrate_weight_sine(bits, freq=freq, nominal_weights=nom)

    assert res["snr_db"] > 50.0
    assert abs(res["weight"][0] / res["weight"][2] - 2.0) < 0.05


def test_osr1_early_stopped_search_matches_static_solve_at_returned_frequency():
    """Final weights/reference at osr=1 must be the exact static fit at refined_frequency."""
    n_samples = 2048
    n = np.arange(n_samples)
    true_freq = 101.3 / n_samples
    bits = sar_convert(
        0.5 + 0.49 * np.sin(2 * np.pi * true_freq * n + 0.4), sar_ideal_weights(12)
    )

    res = calibrate_weight_sine(
        bits, freq=true_freq + 0.25 / n_samples, force_search=True, max_iter=3
    )
    static = calibrate_weight_sine(
        bits, freq=res["refined_frequency"], force_search=False
    )

    np.testing.assert_allclose(res["weight"], static["weight"], rtol=1e-10, atol=1e-12)
    assert res["snr_db"] == pytest.approx(static["snr_db"], abs=1e-9)


def test_osr_gt1_stopband_harmonics_are_excluded_without_blowup():
    """Harmonics at or above 0.5/osr must not destabilize the filtered solve."""
    n_samples = 8192
    n = np.arange(n_samples)
    freq = 0.02219136556  # in-band for osr=16 (cutoff=0.03125), 3*f near stopband null
    bits = sar_convert(
        0.5 + 0.45 * np.sin(2 * np.pi * freq * n + 0.3), sar_ideal_weights(10)
    )

    res1 = calibrate_weight_sine(bits, freq=freq, osr=16, harmonic_order=1)
    with pytest.warns(UserWarning, match="stopband"):
        res3 = calibrate_weight_sine(bits, freq=freq, osr=16, harmonic_order=3)

    assert abs(res3["snr_db"] - res1["snr_db"]) < 0.1
    assert np.max(np.abs(res3["ideal"][0])) < 1.5


def test_python_frequency_policy_handles_constant_dead_msbs():
    """Dead MSBs must not poison coarse frequency estimation under policy='python'."""
    n_samples = 2048
    n = np.arange(n_samples)
    true_freq = 101.0 / n_samples
    bits = sar_convert(
        0.008 + 0.005 * np.sin(2 * np.pi * true_freq * n + 0.3), sar_ideal_weights(12)
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        res = calibrate_weight_sine(bits, freq=None, frequency_policy="python")

    assert abs(res["refined_frequency"] - true_freq) * n_samples < 1e-2
    assert res["snr_db"] > 30.0


def test_calibration_input_validation_guards():
    """Reject non-positive/NaN freq in lite & osr=1, and mismatched target_weights."""
    n = np.arange(256)
    bits = sar_convert(0.5 + 0.4 * np.sin(2 * np.pi * (11 / 256) * n), sar_ideal_weights(6))

    for bad_freq in (0.0, -0.1, np.nan, np.inf, 0.6):
        with pytest.raises(ValueError, match="freq"):
            calibrate_weight_sine_lite(bits, bad_freq)

    for bad_freq in (-0.1, np.nan, np.inf, 0.6):
        with pytest.raises(ValueError, match="freq"):
            calibrate_weight_sine(bits, freq=bad_freq)

    res = calibrate_weight_sine(bits, freq=11 / 256)
    with pytest.raises(ValueError, match="length"):
        scale_calibration_output(res, target_weights=np.ones(4))


def test_dither_hdc_commits_zero_latency_and_matured_updates_before_freeze():
    """controller_latency=0 updates immediately; freezing at commit_at preserves update."""
    rng = np.random.default_rng(0)
    p = rng.choice([-1, 1], size=(256, 3))
    q = np.zeros(256, dtype=int)
    s = np.ones(256, dtype=int)
    rails = np.zeros(256, dtype=bool)
    # Default DitherHDCTracker dither_peak=1/5 -> per-PN step = 1/15
    r_be = np.clip(2.0 * 1.05 * (p.sum(axis=1) / 15.0), -1.0, 1.0)

    tr0 = DitherHDCTracker(fixed=False, window_log2=8, controller_latency=0)
    r1 = tr0.process(r_be, p, q, s, rails, learn=True)
    assert r1["updates"] == 1
    assert r1["pending_update"] is False
    assert r1["coefficients"]["linear"] == pytest.approx(1.0 / 1.05, rel=0.05)

    r2 = tr0.process(r_be, p, q, s, rails, learn=False)
    assert r2["coefficients"]["linear"] == pytest.approx(r1["coefficients"]["linear"])

    # Latency > 0 ending exactly at samples == commit_at must commit when freezing.
    tr_lat = DitherHDCTracker(fixed=False, window_log2=8, controller_latency=16)
    p_long = np.vstack([p, p[:16]])
    q_long = np.zeros(272, dtype=int)
    s_long = np.ones(272, dtype=int)
    rails_long = np.zeros(272, dtype=bool)
    r_long = np.clip(2.0 * 1.05 * (p_long.sum(axis=1) / 15.0), -1.0, 1.0)
    tr_lat.process(r_long, p_long, q_long, s_long, rails_long, learn=True)
    frozen = tr_lat.process(r_be[:16], p[:16], q[:16], s[:16], rails[:16], learn=False)
    assert frozen["coefficients"]["linear"] == pytest.approx(1.0 / 1.05, rel=0.05)
