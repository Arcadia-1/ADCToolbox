"""Regression tests for dout overflow, enob sweep, weight radix, residual scatter, and bit activity."""

import time
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from adctoolbox.dout import (
    analyze_bit_activity,
    analyze_enob_sweep,
    analyze_overflow,
    analyze_weight_radix,
    plot_residual_scatter,
)
from adctoolbox.models.sar import sar_convert, sar_ideal_weights


def test_analyze_overflow_negative_zero_weights_and_ofb_validation():
    bits = np.array([[0, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=float)

    rmin, rmax, _, _ = analyze_overflow(bits, [4.0, 2.0, -0.5], create_plot=False)
    assert np.all(rmin <= rmax)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rmin_z, rmax_z, oz, oo = analyze_overflow(bits, [4.0, 2.0, 0.0], create_plot=False)
    assert not caught
    assert np.isnan(rmin_z[2]) and np.isnan(rmax_z[2])
    assert oz[2] == 0.0 and oo[2] == 0.0

    for bad_ofb in (0, 4, -1):
        with pytest.raises(ValueError, match="ofb"):
            analyze_overflow(bits, [4.0, 2.0, 1.0], ofb=bad_ofb, create_plot=False)


def test_analyze_enob_sweep_constant_msb_plot_and_oversampled_calibration():
    n = np.arange(2048)
    freq = 101.0 / 2048
    bits_const_msb = sar_convert(
        0.25 + 0.2 * np.sin(2 * np.pi * freq * n + 0.3), sar_ideal_weights(8)
    )

    fig, ax = plt.subplots()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        enob_sweep, n_bits_vec = analyze_enob_sweep(
            bits_const_msb, freq=freq, create_plot=True, ax=ax
        )
    assert len(enob_sweep) == 8
    assert np.isfinite(enob_sweep[-1]) and enob_sweep[-1] > 5.5
    plt.close(fig)

    # 2nd-order noise-shaping 6-bit SAR with 4% cap mismatch at OSR=16
    n8 = np.arange(8192)
    f_ns = 23.0 / 8192
    x = 0.5 + 0.35 * np.sin(2 * np.pi * f_ns * n8 + 0.2)
    nom6 = sar_ideal_weights(6)
    rng = np.random.default_rng(7)
    actual6 = nom6 * (1.0 + 0.04 * rng.standard_normal(6))
    actual6 /= actual6.sum()
    q1 = q2 = 0.0
    bits_ns = np.zeros((8192, 6))
    for i in range(8192):
        u = np.clip(x[i] - 2 * q1 + q2, 0.0, 1.0)
        b = sar_convert(np.array([u]), actual6)[0]
        v = float(b @ actual6)
        q2, q1 = q1, v - u
        bits_ns[i] = b

    enob_osr, _ = analyze_enob_sweep(
        bits_ns, freq=f_ns, osr=16, win_type="hann", create_plot=False
    )
    assert enob_osr[-1] > 10.5


def test_analyze_weight_radix_fast_high_bit_subradix_and_zero_guard():
    t0 = time.perf_counter()
    res20 = analyze_weight_radix(2.0 ** np.arange(19, -1, -1), create_plot=False)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.25
    assert res20["wgtsca"] == pytest.approx(1.0, abs=1e-12)
    assert res20["effres"] == pytest.approx(20.0, abs=1e-12)

    res_sub = analyze_weight_radix(1.8 ** np.arange(13, -1, -1), create_plot=False)
    assert res_sub["wgtsca"] == pytest.approx(1.0, rel=1e-3)

    with pytest.raises(ValueError, match="non-zero"):
        analyze_weight_radix(np.zeros(4), create_plot=False)


def test_plot_residual_scatter_column_weights_and_pair_validation():
    bits = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=float)
    w = np.array([4.0, 2.0, 1.0])
    sig = bits @ w + 0.1

    res = plot_residual_scatter(sig, bits, w.reshape(-1, 1), pairs=[(0, 1), (1, 3)], create_plot=False)
    assert res["residuals_y"][0].shape == (4,)
    np.testing.assert_allclose(res["residuals_y"][0], sig - bits[:, :1] @ w[:1])
    np.testing.assert_allclose(res["residuals_y"][1], sig - bits @ w)

    # Empty pairs with create_plot=True must not crash with ZeroDivisionError
    empty = plot_residual_scatter(sig, bits, w, pairs=[], create_plot=True)
    assert empty["residuals_x"] == []

    with pytest.raises(ValueError, match="out of range"):
        plot_residual_scatter(sig, bits, w, pairs=[(-1, 2)], create_plot=False)
    with pytest.raises(ValueError, match="weights length"):
        plot_residual_scatter(sig, bits, np.array([4.0, 2.0]), create_plot=False)
    plt.close("all")


def test_analyze_bit_activity_expands_ylim_for_stuck_bits():
    bits = np.column_stack([np.zeros(100), np.ones(100), np.tile([0, 1], 50)])
    fig, ax = plt.subplots()
    usage = analyze_bit_activity(bits, create_plot=True, ax=ax)
    np.testing.assert_allclose(usage, [0.0, 100.0, 50.0])
    ymin, ymax = ax.get_ylim()
    assert ymin <= 0.0 and ymax >= 100.0
    plt.close(fig)
