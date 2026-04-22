"""Unit tests for calibrate_foreground.

Coverage:
- zero-mismatch identity
- round-trip: inject known mismatch -> calibrate -> re-extract -> residuals ~ 0
- both 'fft' and 'farrow' skew methods
- rejects unknown method names
"""
from __future__ import annotations

import numpy as np
import pytest

from adctoolbox import calibrate_foreground, extract_mismatch_sine


def _synth(M, N, fs, fin, A, gain, offset, skew):
    T = 1.0 / fs
    x = np.empty(N)
    for n in range(N):
        m = n % M
        t = n * T + skew[m]
        x[n] = gain[m] * A * np.cos(2 * np.pi * fin * t) + offset[m]
    return x


def test_calibrate_is_identity_when_all_zero():
    M = 4
    fs = 1e9
    N = 1024
    fin = 17 * fs / N
    x = np.cos(2 * np.pi * fin * np.arange(N) / fs)
    params = {
        "fin": fin, "A": 1.0,
        "gain": np.ones(M), "offset": np.zeros(M), "skew": np.zeros(M),
    }
    y = calibrate_foreground(x, M, params, fs, skew_method="fft")
    np.testing.assert_allclose(y, x, atol=1e-12)


def test_calibrate_removes_mismatch_fft():
    M = 4
    fs = 1e9
    N = 8192
    fin = 37 * fs / N
    A = 0.5
    rng = np.random.default_rng(1)
    true_gain = 1.0 + 0.02 * rng.standard_normal(M)
    true_offset = 0.005 * rng.standard_normal(M)
    true_skew = 3e-12 * rng.standard_normal(M)
    true_skew -= true_skew.mean()

    x = _synth(M, N, fs, fin, A, true_gain, true_offset, true_skew)
    params = extract_mismatch_sine(x, M, fs=fs, fin=fin)
    y = calibrate_foreground(x, M, params, fs, skew_method="fft")

    params2 = extract_mismatch_sine(y, M, fs=fs, fin=fin)
    np.testing.assert_allclose(params2["gain"], np.ones(M), atol=1e-6)
    np.testing.assert_allclose(params2["offset"], np.zeros(M), atol=1e-6)
    np.testing.assert_allclose(params2["skew"], np.zeros(M), atol=1e-15)


def test_calibrate_removes_mismatch_farrow():
    M = 4
    fs = 1e9
    N = 8192
    fin = 37 * fs / N
    A = 0.5
    rng = np.random.default_rng(2)
    true_gain = 1.0 + 0.02 * rng.standard_normal(M)
    true_offset = 0.005 * rng.standard_normal(M)
    true_skew = 3e-12 * rng.standard_normal(M)
    true_skew -= true_skew.mean()

    x = _synth(M, N, fs, fin, A, true_gain, true_offset, true_skew)
    params = extract_mismatch_sine(x, M, fs=fs, fin=fin)
    y = calibrate_foreground(x, M, params, fs, skew_method="farrow", n_taps=9)

    params2 = extract_mismatch_sine(y, M, fs=fs, fin=fin)
    np.testing.assert_allclose(params2["gain"], np.ones(M), atol=1e-4)
    np.testing.assert_allclose(params2["offset"], np.zeros(M), atol=1e-4)
    # 9-tap Lagrange residual skew on a ~3 ps input: under 10 fs is plenty
    assert np.abs(params2["skew"]).max() < 1e-14


def test_calibrate_rejects_bad_method():
    with pytest.raises(ValueError):
        calibrate_foreground(
            np.zeros(16), 4,
            {"gain": np.ones(4), "offset": np.zeros(4), "skew": np.ones(4) * 1e-12},
            fs=1e9, skew_method="wibble",
        )
