"""Regression tests for aout, fundamentals, siggen, models, and utils audit fixes."""

import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from adctoolbox.aout import (
    analyze_error_autocorr,
    analyze_error_pdf,
    analyze_error_phase_plane,
    analyze_phase_plane,
    compute_inl_from_sine,
    decompose_harmonic_error,
    fit_static_nonlin,
    plot_error_autocorr,
    rearrange_error_by_phase,
    rearrange_error_by_value,
)
from adctoolbox.aout._errsin_compat import errsin_compat
from adctoolbox.common import nsd_to_snr as common_nsd_to_snr, snr_to_nsd as common_snr_to_nsd
from adctoolbox.fundamentals import (
    amplitudes_to_snr,
    fit_sine_4param,
    fold_bin_to_nyquist,
    fold_frequency_to_nyquist,
    nsd_to_snr,
    snr_to_nsd,
)
from adctoolbox.fundamentals.units import (
    nsd_to_snr as units_nsd_to_snr,
    snr_to_nsd as units_snr_to_nsd,
)
from adctoolbox.fundamentals.validate import validate_aout_data
from adctoolbox.models.sar import sar_ideal_weights
from adctoolbox.siggen import ADC_Signal_Generator
from adctoolbox.utils.vpp_for_target_dbfs import vpp_for_target_dbfs


def test_fit_sine_4param_large_scale_svd_and_near_nyquist():
    # Large A * N^2 previously suffered SVD truncation in lstsq
    n_big, amp_big = 2**17, float(2**20)
    f_big = 123.37 / n_big
    y_big = amp_big * np.cos(2 * np.pi * f_big * np.arange(n_big) + 0.3) + amp_big
    res_big = fit_sine_4param(y_big)
    assert res_big["converged"] is True
    assert res_big["rmse"] / amp_big < 1e-11

    # Near-Nyquist tone in (N/2 - 0.5, N/2) previously deadlocked at 0.5
    n_nq = 1024
    f_nq = 511.7 / n_nq
    y_nq = 0.9 * np.cos(2 * np.pi * f_nq * np.arange(n_nq) + 0.4) + 0.1
    res_nq = fit_sine_4param(y_nq)
    assert res_nq["converged"] is True
    assert res_nq["frequency"] == pytest.approx(f_nq, abs=1e-9)
    assert res_nq["rmse"] < 1e-11


def test_decompose_harmonic_error_phase_sign_rms_and_aliasing_collision():
    t = np.arange(4096)
    f = 37.0 / 4096

    # 1) Relative harmonic phase sign
    sig_p = 0.5 * np.cos(2 * np.pi * f * t) + 0.05 * np.cos(2 * np.pi * 2 * f * t + np.pi / 2)
    res_p = decompose_harmonic_error(sig_p, n_harmonics=3, frequency=f, max_iterations=0)
    assert res_p["phases"][1] == pytest.approx(np.pi / 2, abs=1e-9)

    # 2) residual_rms is true RMS (not inflated by 2*sqrt(2))
    noise = 1e-3 * np.random.default_rng(0).standard_normal(4096)
    res_n = decompose_harmonic_error(
        0.5 * np.cos(2 * np.pi * f * t) + noise, n_harmonics=3, frequency=f, max_iterations=0
    )
    true_rms = float(np.sqrt(np.mean(noise**2)))
    assert res_n["residual_rms"] == pytest.approx(true_rms, rel=0.05)

    # 3) Aliasing collision on pure sine at f=0.25 must not fabricate harmonics
    res_c = decompose_harmonic_error(
        0.5 * np.sin(2 * np.pi * 0.25 * t), n_harmonics=5, frequency=0.25, max_iterations=0
    )
    assert res_c["magnitudes"][0] == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(res_c["magnitudes"][1:], 0.0, atol=1e-12)
    assert np.max(np.abs(res_c["harmonic_signal"])) < 1e-12


def test_compute_inl_from_sine_bipolar_voltage_and_validation():
    gen = ADC_Signal_Generator(N=2**16, Fs=float(2**16), Fin=1237.0, A=0.49, DC=0.0)
    sig_q = gen.apply_quantization_noise(n_bits=10, quant_range=(-0.5, 0.5))

    for fs_arg in (None, 1.0, (-0.5, 0.5)):
        inl, dnl, codes = compute_inl_from_sine(sig_q, num_bits=10, full_scale=fs_arg)
        assert len(codes) > 900
        assert np.max(np.abs(dnl)) < 0.05
        assert np.max(np.abs(inl)) < 0.2

    for bad_clip in (-0.01, 0.5, 0.8):
        with pytest.raises(ValueError, match="clip_percent"):
            compute_inl_from_sine(sig_q, num_bits=10, clip_percent=bad_clip)


def test_errsin_compat_rising_falling_edges_match_cosine_slope():
    t = np.arange(4096)
    f = 17.0 / 4096
    # fitted - sig = -0.01*sin(2*theta): positive on rising edge (270..360 deg),
    # negative on falling edge (0..90 deg) for value > 0.
    sig = np.cos(2 * np.pi * f * t) + 0.01 * np.sin(2 * np.pi * 2 * f * t)
    r = errsin_compat(sig, bins=20, freq=f, xaxis="value", disp=False)
    pos = r["xx"] > 0.2
    assert np.nanmean(r["emean_rise"][pos]) > 0.005
    assert np.nanmean(r["emean_fall"][pos]) < -0.005


def test_rearrange_error_by_phase_and_value_exclude_deterministic_distortion():
    t = np.arange(2**14)
    f = 137.0 / 2**14
    # Completely noiseless signal with deterministic HD2 & HD4
    sig = 0.5 * np.cos(2 * np.pi * f * t) + 0.01 * np.cos(2 * np.pi * 2 * f * t) - 0.005 * np.cos(2 * np.pi * 4 * f * t)
    rp = rearrange_error_by_phase(sig, norm_freq=f, n_bins=200, max_iterations=0)
    assert rp["pm_noise_rms_v"] < 1e-3
    assert rp["am_noise_rms_v"] < 1e-3
    assert rp["base_noise_rms_v"] < 1e-3

    # Pure static HD2 without noise: value-binned noise RMS must be ~0 (not |mean|)
    sig_hd2 = 0.5 * np.cos(2 * np.pi * f * t) + 0.02 * (0.5 * np.cos(2 * np.pi * f * t)) ** 2
    rv = rearrange_error_by_value(sig_hd2, norm_freq=f, n_bins=100, max_iterations=0)
    assert np.nanmedian(rv["error_rms"]) < 1e-3


def test_analyze_error_pdf_kl_divergence_is_nonnegative():
    n = 4096
    f = 137.0 / n
    sig = 0.5 + 0.49 * np.sin(2 * np.pi * f * np.arange(n)) + 1e-5 * np.random.default_rng(42).standard_normal(n)
    res = analyze_error_pdf(sig, resolution=8, full_scale=1.0, frequency=f, max_iterations=0, create_plot=False)
    assert res["kl_divergence"] >= 0.0


def test_phase_plane_fixes_eiv_lag_glitch_and_create_plot_false():
    # 1) No errors-in-variables cubic artifact on pure linear thermal noise
    n = 16384
    sig_lin = 0.5 * np.sin(2 * np.pi * (137 / n) * np.arange(n)) + 0.05 * np.random.default_rng(0).standard_normal(n)
    fig, ax = plt.subplots()
    res_epp = analyze_error_phase_plane(sig_lin, ax=ax, create_plot=False)
    assert abs(res_epp["trend_coeffs"][0]) < 0.05
    assert len(ax.lines) == 0 and len(ax.collections) == 0
    plt.close(fig)

    # 2) Near-Nyquist lag search reaches 90 deg (k=50 for f=0.495)
    res_nq = analyze_phase_plane(
        0.5 * np.sin(2 * np.pi * 0.495 * np.arange(4096)), lag="auto", create_plot=False
    )
    assert abs(res_nq["lag"] - 50) <= 1

    # 3) Single glitch reported once, create_plot=False respected, lag<=0 raises
    s_gl = 0.5 * np.sin(2 * np.pi * 0.25 * np.arange(1024))
    s_gl[500] += 0.2
    fig2, ax2 = plt.subplots()
    res_gl = analyze_phase_plane(s_gl, lag="auto", ax=ax2, create_plot=False)
    np.testing.assert_array_equal(res_gl["outliers"], [500])
    assert len(ax2.collections) == 0
    plt.close(fig2)

    with pytest.raises(ValueError, match="Lag"):
        analyze_phase_plane(s_gl, lag=0, create_plot=False)


def test_plot_error_autocorr_does_not_refit_sine_and_guards_max_lag():
    err = 0.01 * np.sin(2 * np.pi * 0.1 * np.arange(1024)) + 1e-3 * np.random.default_rng(0).standard_normal(1024)
    acf, lags = plot_error_autocorr(err, max_lag=10, normalize=True, create_plot=False)
    assert acf[lags == 1][0] > 0.75

    with pytest.raises(ValueError, match="max_lag"):
        analyze_error_autocorr(np.sin(np.arange(50)), max_lag=60, create_plot=False)


def test_utils_units_snr_nsd_validate_sar_and_list_input():
    # vpp_for_target_dbfs import and math
    assert vpp_for_target_dbfs(1.0, -6.5, -0.5) == pytest.approx(10 ** (6.0 / 20), rel=1e-12)

    # Unified snr_to_nsd / nsd_to_snr signatures across namespaces
    ref = snr_to_nsd(80, 1e6, 256, -6)
    assert units_snr_to_nsd(80, 1e6, 256, -6) == pytest.approx(ref)
    assert common_snr_to_nsd(80, 1e6, 256, -6) == pytest.approx(ref)
    assert units_snr_to_nsd(80, 1e6, osr=256, signal_pwr_dbfs=-6) == pytest.approx(ref)
    assert units_nsd_to_snr(ref, 1e6, 256, -6) == pytest.approx(80.0, abs=1e-12)
    assert common_nsd_to_snr(ref, 1e6, 256, -6) == pytest.approx(80.0, abs=1e-12)

    # amplitudes_to_snr(0, 0) -> NaN, osr <= 0 -> ValueError
    assert np.isnan(amplitudes_to_snr(0.0, 0.0))
    with pytest.raises(ValueError, match="osr"):
        amplitudes_to_snr(1.0, 0.01, osr=-4)

    # fold_bin_to_nyquist accepts arrays; fold_frequency_to_nyquist checks fs > 0
    np.testing.assert_allclose(fold_bin_to_nyquist(np.array([10, 600, -10]), 1024), [10.0, 424.0, 10.0])
    with pytest.raises(ValueError, match="fs"):
        fold_frequency_to_nyquist(10.0, 0.0)

    # validate_aout_data checks axis-0 sample count
    with pytest.raises(ValueError, match="Insufficient"):
        validate_aout_data(np.random.default_rng(0).standard_normal((10, 200)), min_samples=100)

    # sar_ideal_weights rejects negative redundant_bit
    with pytest.raises(ValueError, match="redundant_bit"):
        sar_ideal_weights(4, redundant_bit=-1)

    # fit_static_nonlin accepts plain Python list
    sig_list = (0.5 * np.sin(2 * np.pi * (17 / 512) * np.arange(512))).tolist()
    k2, k3, _, _ = fit_static_nonlin(sig_list, order=3)
    assert abs(k2) < 1e-6 and abs(k3) < 1e-6
