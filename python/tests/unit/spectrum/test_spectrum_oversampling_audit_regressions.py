"""Regression tests for spectrum and oversampling audit fixes."""

import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from adctoolbox.oversampling import ntf_analyzer, ntfperf, perfosr
from adctoolbox.spectrum import compute_spectrum, plot_spectrum_polar, quick_sndr
from adctoolbox.spectrum.sweep_performance_vs_osr import sweep_performance_vs_osr


def test_auto_side_bin_near_dc_does_not_swallow_band_and_handles_ultra_low_floor():
    n = 4096
    t = np.arange(n)
    noise = 1e-4 * np.random.default_rng(42).standard_normal(n)
    # Slow-decay window (hamming) near DC must fall back cleanly instead of
    # swallowing the whole Nyquist band past DC.
    x_low = 0.9 * np.sin(2 * np.pi * 10.3 * t / n) + noise
    res_ham = compute_spectrum(x_low, win_type="hamming", max_scale_range=1.0)
    assert res_ham["plot_data"]["side_bin"] <= 10
    assert res_ham["metrics"]["enob"] < 13.0

    # Ultra-low noise floor below 1e-20 power (~5e-24/bin) must still be
    # tracked cleanly (positive bins > 0.0 instead of > 1e-20).
    x_quiet = 1e-6 * np.sin(2 * np.pi * 512.37 * t / n) + 1e-10 * np.random.default_rng(7).standard_normal(n)
    res_kaiser = compute_spectrum(x_quiet, win_type="kaiser", max_scale_range=1.0)
    assert res_kaiser["metrics"]["sndr_dbc"] == pytest.approx(77.0, abs=2.0)
    assert res_kaiser["plot_data"]["side_bin"] < 30


def test_kaiser_noise_free_auto_and_quick_sndr():
    n = 4096
    t = np.arange(n)
    x_nc = 0.5 * np.sin(2 * np.pi * 1024.37 * t / n)
    r_nc = compute_spectrum(x_nc, max_scale_range=1.0, win_type="kaiser")
    assert r_nc["metrics"]["sndr_dbc"] > 180.0
    assert r_nc["metrics"]["sfdr_dbc"] > 180.0

    x_c = 0.5 * np.sin(2 * np.pi * 123 * t / n)
    r_c = compute_spectrum(x_c, max_scale_range=1.0, win_type="kaiser")
    q_c = quick_sndr(x_c, win_type="kaiser")
    assert r_c["metrics"]["sndr_dbc"] > 200.0
    assert q_c["sndr_dbc"] > 135.0


def test_high_dynamic_range_does_not_clip_at_minus_150_dbfs():
    n = 32768
    t = np.arange(n)
    amp = 0.1  # -20 dBFS wrt [-1, 1]
    noise_rms = (amp / np.sqrt(2)) * 10 ** (-140 / 20)
    x = amp * np.sin(2 * np.pi * 123 * t / n) + noise_rms * np.random.default_rng(1).standard_normal(n)
    res = compute_spectrum(x, max_scale_range=[-1, 1], win_type="hann", side_bin=1)
    assert res["metrics"]["sndr_dbc"] == pytest.approx(140.0, abs=0.5)
    assert res["metrics"]["snr_dbc"] == pytest.approx(140.0, abs=0.5)
    assert res["metrics"]["noise_floor_dbfs"] == pytest.approx(-160.0, abs=0.5)


def test_nf_method3_warns_and_returns_nan_when_all_bins_masked():
    n = 512
    t = np.arange(n)
    x = 0.5 * np.sin(2 * np.pi * 5 * t / n) + 2.5e-4 * np.random.default_rng(0).standard_normal(n)
    with pytest.warns(RuntimeWarning, match="No noise bins remain"):
        res = compute_spectrum(
            x, osr=16, nf_method=3, win_type="blackmanharris", side_bin=3, max_harmonic=5
        )
    assert np.isnan(res["metrics"]["snr_dbc"])
    assert np.isnan(res["metrics"]["noise_floor_dbfs"])


def test_coherent_average_preserves_adjacent_hd3_when_hd2_is_zero():
    n, m = 1024, 64
    rng = np.random.default_rng(0)
    phases = rng.uniform(0, 2 * np.pi, m)
    t = np.arange(n)[None, :]
    theta = 2 * np.pi * 205 * t / n + phases[:, None]  # H2=410, H3=409
    x = 0.5 * np.cos(theta) + 0.5 * 10 ** (-50 / 20) * np.cos(3 * theta + 0.7) + 1e-6 * rng.standard_normal((m, n))
    rp = compute_spectrum(x, win_type="blackmanharris", coherent_averaging=False, max_harmonic=5)
    rc = compute_spectrum(x, win_type="blackmanharris", coherent_averaging=True, max_harmonic=5)
    # H3 center (409) is flanked by H2 (410) and H8 (408); preserving the center
    # bin keeps >= 50% (-3 dB) of BH lobe power instead of losing -21 dB.
    assert abs(rc["metrics"]["harmonics_dbc"][1] - rp["metrics"]["harmonics_dbc"][1]) < 3.5
    rp_h = compute_spectrum(x, win_type="hann", coherent_averaging=False, max_harmonic=5)
    rc_h = compute_spectrum(x, win_type="hann", coherent_averaging=True, max_harmonic=5)
    assert abs(rc_h["metrics"]["harmonics_dbc"][1] - rp_h["metrics"]["harmonics_dbc"][1]) < 2.0


def test_side_bin_zero_masks_dc_and_bin1_does_not_bias_harmonics():
    n = 1024
    t = np.arange(n)
    x = 0.5 * np.cos(2 * np.pi * 1 * t / n)
    r0 = compute_spectrum(x, win_type="hann", side_bin=0, max_scale_range=1.0)
    assert r0["plot_data"]["spur_bin_idx"] != 0

    x_n = x + 1e-4 * np.random.default_rng(0).standard_normal(n)
    r1 = compute_spectrum(x_n, win_type="hann", max_scale_range=1.0, max_harmonic=5)
    assert r1["plot_data"]["fundamental_bin_fractional"] == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_array_equal(r1["plot_data"]["harmonic_bins"], [2, 3, 4, 5])


def test_unclipped_osr_spur_not_inflated_and_collided_harmonic_detected():
    n = 1024
    t = np.arange(n)
    sig_sp = np.sin(2 * np.pi * 100.5 * t / n) + 1e-3 * np.sin(2 * np.pi * 241 * t / n)
    r_sp = compute_spectrum(sig_sp, osr=2, win_type="hann", side_bin=15, max_scale_range=[-1, 1])
    assert r_sp["metrics"]["sfdr_dbc"] == pytest.approx(60.0, abs=0.1)

    # HD2 at 342 lies inside fundamental lobe [340..342]
    sig_col = np.sin(2 * np.pi * 341 * t / n) + 1e-3 * np.sin(2 * np.pi * 2 * 341 * t / n)
    r_col = compute_spectrum(sig_col, win_type="hann", side_bin=1, max_scale_range=[-1, 1])
    assert 2 in r_col["plot_data"]["collided_harmonics"]


def test_spectrum_input_edge_cases_and_rect_alias():
    n = 1024
    t = np.arange(n)
    s = 0.5 * np.sin(2 * np.pi * 13 * t / n)

    # 2D captures with inter-run DC drift
    r2d = compute_spectrum(np.vstack([s + 10.0, s - 10.0]))
    assert r2d["metrics"]["sig_pwr_dbfs"] == pytest.approx(0.0, abs=0.05)

    # All-zero input produces NaN cleanly without +inf or divide-by-zero warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rz = compute_spectrum(np.zeros(n))
    assert not caught
    assert np.isnan(rz["metrics"]["noise_floor_dbfs"])
    assert np.isnan(rz["metrics"]["sndr_dbc"])

    # 'rect' alias matches 'rectangular'
    r_rect = compute_spectrum(s, win_type="rect")
    r_box = compute_spectrum(s, win_type="rectangular")
    assert r_rect["plot_data"]["side_bin"] == 0
    assert r_rect["metrics"]["sndr_dbc"] == pytest.approx(r_box["metrics"]["sndr_dbc"], abs=1e-9)

    with pytest.warns(UserWarning, match="Unrecognized win_type"):
        compute_spectrum(s, win_type="unknown_window_xyz")

    with pytest.raises(ValueError, match="osr"):
        compute_spectrum(s, osr=0.5)

    # max_harmonic=0 should not crash
    rh0 = compute_spectrum(s, max_harmonic=0)
    assert rh0["metrics"]["harmonics_dbc"].size == 0


def test_sweep_performance_vs_osr_dsm_sfdr_scalar_and_dc_harmonic_plot():
    # 1) Coherent -60 dBc HD2: SFDR should be 60.0 dB (not +1.76 dB biased)
    n = 4096
    t = np.arange(n)
    x_hd2 = 0.5 * np.sin(2 * np.pi * 37 * t / n) + 0.5e-3 * np.sin(2 * np.pi * 74 * t / n)
    r_hd2 = sweep_performance_vs_osr(x_hd2, osr=[1, 2], create_plot=False)
    assert r_hd2["sfdr"][0] == pytest.approx(60.0, abs=0.05)

    # 2) Scalar osr works in both sweep_performance_vs_osr and perfosr
    osr_out, sndr_out, sfdr_out, enob_out = perfosr(x_hd2, osr=4)
    assert osr_out.shape == (1,) and np.isfinite(sndr_out[0])

    # 3) 3rd-order DSM at OSR=64 matches compute_spectrum(blackmanharris)
    n16 = 16384
    t16 = np.arange(n16)
    x_in = 0.5 * np.sin(2 * np.pi * 31 * t16 / n16)
    lsb = 1.0 / 1024
    q = np.zeros(n16)
    y = np.zeros(n16)
    for i in range(n16):
        fb = (3 * q[i - 1] if i >= 1 else 0) - (3 * q[i - 2] if i >= 2 else 0) + (q[i - 3] if i >= 3 else 0)
        u = x_in[i] - fb
        yq = np.round(u / lsb) * lsb
        q[i] = yq - u
        y[i] = yq
    cs = compute_spectrum(y, osr=64, win_type="blackmanharris")
    po = sweep_performance_vs_osr(y, osr=[1, 8, 32, 64], create_plot=False)
    assert abs(po["sndr"][-1] - cs["metrics"]["sndr_dbc"]) < 2.0

    # 4) Harmonic aliasing to DC (f=0.25 -> H4=0) must not crash on plot
    sweep_performance_vs_osr(0.5 * np.sin(2 * np.pi * 0.25 * np.arange(1024)), harmonic=5, create_plot=True)
    plt.close("all")


def test_ntf_analyzer_bounds_accuracy_and_clean_plotting():
    # Exact 0.0 dB for flat NTF=1 even on small grid
    assert ntf_analyzer(([1], [1]), 0.0, 0.5, n_grid=100) == pytest.approx(0.0, abs=1e-12)

    # Exact closed-form 1st-order & 2nd-order at OSR=16
    assert ntf_analyzer(([1, -1], [1, 0]), 0.0, 0.5 / 16) == pytest.approx(30.985, abs=0.05)
    assert ntf_analyzer(([1, -2, 1], [1, 0, 0]), 0.0, 0.5 / 16) == pytest.approx(47.345, abs=0.05)

    # Reject invalid bounds
    for fl, fh in ((0.0, 1.0), (0.3, 0.2), (-0.1, 0.2), (0.2, 0.2)):
        with pytest.raises(ValueError, match="Frequency bounds"):
            ntfperf(([1, -1], [1, 0]), fl, fh)

    # Plotting must not emit log10(0) or Agg non-interactive warnings, and marker lines must be finite
    fig = plt.figure()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ntf_analyzer(([1, -1], [1, 0]), 0.0, 0.5 / 16, is_plot=1)
    assert not caught
    for line in plt.gca().lines:
        assert np.all(np.isfinite(line.get_ydata()))
    plt.close(fig)


def test_plot_spectrum_polar_radial_ticks_align_with_plotted_radii():
    n = 1024
    t = np.arange(n)
    x = 0.5 * np.sin(2 * np.pi * 37 * t / n)  # -6.02 dBFS wrt [-1, 1]
    res = compute_spectrum(x, max_scale_range=[-1, 1], win_type="boxcar", coherent_averaging=True)
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    plot_spectrum_polar(res, fixed_radial_range=105, ax=ax)
    ticks = ax.get_yticks()
    labels = [int(lbl.get_text()) for lbl in ax.get_yticklabels()]
    # Every ring labeled L dB must sit at radius L - (-105) = L + 105
    for r_tick, db_label in zip(ticks, labels):
        assert r_tick == pytest.approx(db_label + 105, abs=1e-9)
    plt.close(fig)
