"""Regression tests for timeinterleave, toolset dashboards, and compare utilities."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from adctoolbox.spectrum import compute_spectrum
from adctoolbox.timeinterleave import (
    calibrate_foreground,
    extract_mismatch_sine,
    fractional_delay_farrow,
    predict_spurs,
)
from adctoolbox.toolset.generate_aout_dashboard import generate_aout_dashboard as dash_2x4
from adctoolbox.toolset.generate_aout_dashboard_3x4 import generate_aout_dashboard as dash_3x4


def test_extract_mismatch_sine_exact_on_noncoherent_tone_and_guards_aliases():
    m, n, fs = 4, 4096, 1e9
    fin = 37.3 * fs / n  # non-coherent
    idx = np.arange(n)
    x_clean = 0.5 * np.cos(2 * np.pi * fin * idx / fs + 0.25)

    est = extract_mismatch_sine(x_clean, m, fs=fs, fin=fin)
    np.testing.assert_allclose(est["gain"], np.ones(m), atol=1e-12)
    np.testing.assert_allclose(est["offset"], np.zeros(m), atol=1e-12)
    np.testing.assert_allclose(est["skew"], np.zeros(m), atol=1e-20)

    # Non-coherent with real mismatch recovers exact parameters
    gain = np.array([1.02, 0.98, 1.01, 0.99])
    offset = np.array([0.005, -0.004, 0.002, -0.003])
    skew = np.array([3e-12, -2e-12, 1e-12, -2e-12])
    x_mis = gain[idx % m] * 0.5 * np.cos(2 * np.pi * fin * (idx / fs + skew[idx % m]) + 0.25) + offset[idx % m]
    est_mis = extract_mismatch_sine(x_mis, m, fs=fs, fin=fin)
    np.testing.assert_allclose(est_mis["gain"], gain / gain.mean(), atol=1e-12)
    np.testing.assert_allclose(est_mis["offset"], offset, atol=1e-12)
    np.testing.assert_allclose(est_mis["skew"], skew, atol=1e-20)

    # Sub-ADC alias frequencies fs/(2M), fs/M, fin<=0, N==0 raise ValueError
    for bad_fin in (fs / (2 * m), fs / m, 0.0, -1e6):
        with pytest.raises(ValueError):
            extract_mismatch_sine(x_clean, m, fs=fs, fin=bad_fin)
    with pytest.raises(ValueError, match="empty"):
        extract_mismatch_sine(np.array([]), m, fs=fs, fin=fin)


def test_predict_spurs_nyquist_offset_and_skew_carrier_attenuation():
    fs = 1e9
    n = 8192
    idx = np.arange(n)

    # 1) Nyquist offset spur (k = M/2) matches compute_spectrum dBc / dBFS to 0.01 dB
    fin_c = 37 * fs / n
    off_nyq = np.array([0.01, -0.01, 0.01, -0.01])
    x_nyq = 0.5 * np.cos(2 * np.pi * fin_c * idx / fs) + off_nyq[idx % 4]
    spurs = predict_spurs(
        {"gain": [1, 1, 1, 1], "offset": off_nyq, "skew": [0, 0, 0, 0], "fin": fin_c, "A": 0.5},
        fs=fs,
        full_scale=1.0,
    )
    nyq = [s for s in spurs if s["amp"] > 0][0]
    spec = compute_spectrum(x_nyq, fs=fs, max_scale_range=[-1, 1], win_type="boxcar")
    assert nyq["dbc"] == pytest.approx(-spec["metrics"]["sfdr_dbc"], abs=0.01)

    # 2) Large skew carrier attenuation reflected in fundamental_amp / dbc
    fin_hi = 3957 * fs / n
    skew_big = np.array([0.25 / fs, -0.25 / fs])
    x_sk = 0.5 * np.cos(2 * np.pi * fin_hi * (idx / fs + skew_big[idx % 2]))
    sp_sk = predict_spurs(
        {"gain": [1, 1], "offset": [0, 0], "skew": skew_big, "fin": fin_hi, "A": 0.5},
        fs=fs,
        full_scale=1.0,
    )
    spec_sk = compute_spectrum(x_sk, fs=fs, max_scale_range=[-1, 1], win_type="boxcar")
    assert sp_sk[0]["dbc"] == pytest.approx(-spec_sk["metrics"]["sfdr_dbc"], abs=0.02)


def test_calibrate_foreground_exact_fft_at_large_skew_and_validates_gain():
    # Large skew (0.55 samples) previously caused Richardson iteration to diverge
    n = 1024
    idx = np.arange(n)
    sk = np.array([0.55, -0.55])
    x = 0.5 * np.cos(2 * np.pi * (461 / n) * (idx + sk[idx % 2]))
    x_ideal = 0.5 * np.cos(2 * np.pi * (461 / n) * idx)

    y = calibrate_foreground(x, 2, {"gain": [1, 1], "offset": [0, 0], "skew": sk}, fs=1.0, skew_method="fft")
    np.testing.assert_allclose(y, x_ideal, atol=1e-10)

    # Zero or negative gain must raise ValueError rather than producing NaNs
    for bad_gain in ([1.0, 0.0], [1.0, -0.5]):
        with pytest.raises(ValueError, match="gain"):
            calibrate_foreground(
                x, 2, {"gain": bad_gain, "offset": [0, 0], "skew": [0, 0]}, fs=1.0
            )


def test_fractional_delay_farrow_preserves_boundary_samples_and_continuity():
    # Linear ramp shifted by +0.6: y[17] = x(16.4) uses x[13..19], all in-bounds
    x = np.arange(1.0, 21.0)
    y = fractional_delay_farrow(x, delay_sec=0.6, fs=1.0, n_taps=7)
    assert y[17] == pytest.approx(17.4, abs=1e-12)

    # Continuous across |D| = N - 0.5 (no snap to all zeros at D = 9.6 on len 10)
    x10 = np.arange(1.0, 11.0)
    y94 = fractional_delay_farrow(x10, delay_sec=9.4, fs=1.0, n_taps=7)
    y96 = fractional_delay_farrow(x10, delay_sec=9.6, fs=1.0, n_taps=7)
    assert np.any(y96 != 0.0)
    assert np.max(np.abs(y96 - y94)) < 5.0


def test_aout_dashboards_pass_freq_and_respect_show(tmp_path):
    n = 1024
    fs = 100e6
    fin = 37 * fs / n
    sig = 0.5 + 0.45 * np.sin(2 * np.pi * fin * np.arange(n) / fs) + 1e-4 * np.random.default_rng(0).standard_normal(n)

    out3 = tmp_path / "dash3x4.png"
    fig3, axes3 = dash_3x4(sig, fs=fs, freq=fin, output_path=out3, show=False)
    assert out3.exists() and len(axes3) == 12

    out2 = tmp_path / "dash2x4.png"
    fig2, axes2 = dash_2x4(sig, fs=fs, freq=fin, output_path=out2)
    assert out2.exists() and len(axes2) == 8
    plt.close("all")


def test_csv_comparator_shape_mismatch_and_matching_infs(tmp_path):
    from tests.compare._csv_comparator import CSVComparator

    cmp = CSVComparator(threshold=1e-6)
    f_a = tmp_path / "a.csv"
    f_b = tmp_path / "b.csv"

    # Shape mismatch must return structured ERROR without crashing on _build_result
    np.savetxt(f_a, [1.0, 2.0, 3.0], delimiter=",")
    np.savetxt(f_b, [1.0, 2.0], delimiter=",")
    res_shape = cmp.compare_pair(f_a, f_b)
    assert res_shape["status"] == "ERROR"
    assert "Shape mismatch" in res_shape["msg"]

    # Matching +/-inf should pass as PERFECT rather than nan failure
    np.savetxt(f_a, [np.inf, -np.inf, 1.0], delimiter=",")
    np.savetxt(f_b, [np.inf, -np.inf, 1.0], delimiter=",")
    res_inf = cmp.compare_pair(f_a, f_b)
    assert res_inf["status"] == "PERFECT"
    assert res_inf["max_diff_abs"] == 0.0
