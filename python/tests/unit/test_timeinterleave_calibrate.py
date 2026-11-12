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


# ---------------------------------------------------------------------------
# Tones above fs/(2M)
# ---------------------------------------------------------------------------
# A sub-ADC samples at fs/M, so a tone above fs/(2M) is aliased inside each
# channel. The skew used to be corrected with a delay at the channel rate,
# which shifts the alias instead of the signal: SFDR stayed at 58-75 dB for
# 1 ps of skew, where the offset/gain-only result is 56-61 dB.

def _mismatched_capture(M, tone_bin, N=8192, fs=1e9, seed=1):
    rng = np.random.default_rng(seed)
    gain = 1.0 + 0.01 * rng.standard_normal(M)
    offset = 0.005 * rng.standard_normal(M)
    skew = 1e-12 * rng.standard_normal(M)
    skew -= skew.mean()
    fin = tone_bin * fs / N
    n = np.arange(N)
    m = n % M
    x = gain[m] * 0.5 * np.cos(2 * np.pi * fin * (n / fs + skew[m])) + offset[m]
    return x, fs, fin


def _sfdr_db(x):
    power = np.abs(np.fft.rfft(x - np.mean(x))) ** 2
    carrier = int(np.argmax(power))
    carrier_power = power[carrier]
    power[carrier] = 0.0
    return 10 * np.log10(carrier_power / max(power.max(), 1e-300))


@pytest.mark.parametrize(
    "M, tone_bin",
    [
        (4, 37),      # below fs/(2M): already exact before
        (8, 301),
        (4, 1499),    # 0.183 fs, above fs/8
        (2, 3957),    # 0.483 fs, above fs/4
        (8, 3957),
        (4, 4000),    # 0.488 fs
    ],
)
def test_fft_skew_correction_is_exact_across_the_first_nyquist_zone(M, tone_bin):
    x, fs, fin = _mismatched_capture(M, tone_bin)
    assert _sfdr_db(x) < 50

    params = extract_mismatch_sine(x, M, fs=fs, fin=fin)
    y = calibrate_foreground(x, M, params, fs, skew_method="fft")

    assert _sfdr_db(y) > 200
    residual = extract_mismatch_sine(y, M, fs=fs, fin=fin)
    np.testing.assert_allclose(residual["gain"], np.ones(M), atol=1e-9)
    np.testing.assert_allclose(residual["offset"], np.zeros(M), atol=1e-9)
    np.testing.assert_allclose(residual["skew"], np.zeros(M), atol=1e-18)


@pytest.mark.parametrize("M, tone_bin, floor_db", [(4, 37, 130), (8, 301, 120), (4, 1499, 100)])
def test_farrow_skew_correction_above_the_channel_nyquist(M, tone_bin, floor_db):
    x, fs, fin = _mismatched_capture(M, tone_bin)

    params = extract_mismatch_sine(x, M, fs=fs, fin=fin)
    y = calibrate_foreground(x, M, params, fs, skew_method="farrow")

    assert _sfdr_db(y) > floor_db


@pytest.mark.parametrize("M, tone_bin", [(2, 3957), (8, 3957)])
def test_farrow_near_nyquist_still_removes_gain_and_offset(M, tone_bin):
    # A 7-tap Lagrange filter cannot interpolate at 0.48 fs; the skew spur
    # stays near 60 dB but the correction must not make anything worse.
    x, fs, fin = _mismatched_capture(M, tone_bin)

    params = extract_mismatch_sine(x, M, fs=fs, fin=fin)
    y = calibrate_foreground(x, M, params, fs, skew_method="farrow")

    assert _sfdr_db(y) > _sfdr_db(x) + 10
    residual = extract_mismatch_sine(y, M, fs=fs, fin=fin)
    np.testing.assert_allclose(residual["gain"], np.ones(M), atol=1e-3)
    np.testing.assert_allclose(residual["offset"], np.zeros(M), atol=1e-6)


def test_calibration_leaves_a_mismatch_free_capture_alone_at_high_frequency():
    M, N, fs = 4, 4096, 1e9
    x = 0.5 * np.cos(2 * np.pi * 1999 * np.arange(N) / N)
    params = extract_mismatch_sine(x, M, fs=fs, fin=1999 * fs / N)

    y = calibrate_foreground(x, M, params, fs)

    np.testing.assert_allclose(y, x, atol=1e-9)


def test_tone_above_the_aggregate_nyquist_warns_and_still_fixes_gain_and_offset():
    M, N, fs = 4, 4096, 1e9
    fin = (N + 301) * fs / N        # second Nyquist zone, aliases to bin 301
    rng = np.random.default_rng(2)
    gain = 1.0 + 0.01 * rng.standard_normal(M)
    offset = 0.005 * rng.standard_normal(M)
    skew = np.array([0.4e-12, -0.3e-12, 0.1e-12, -0.2e-12])
    n = np.arange(N)
    m = n % M
    x = gain[m] * 0.5 * np.cos(2 * np.pi * fin * (n / fs + skew[m])) + offset[m]
    params = extract_mismatch_sine(x, M, fs=fs, fin=fin)

    with pytest.warns(UserWarning, match="above fs/2"):
        y = calibrate_foreground(x, M, params, fs)

    residual = extract_mismatch_sine(y, M, fs=fs, fin=fin)
    np.testing.assert_allclose(residual["gain"], np.ones(M), atol=1e-4)
    np.testing.assert_allclose(residual["offset"], np.zeros(M), atol=1e-6)
