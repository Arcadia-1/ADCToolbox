"""Unit tests for the fractional_delay_fft and fractional_delay_farrow primitives.

Both functions implement ``y(t) = x(t - delay_sec)`` with the same sign
convention; these tests verify:

- zero delay → identity
- integer sample delay → matches ``np.roll`` (FFT) / zero-padded shift (Farrow)
- half-sample delay of a well-bandlimited tone → analytic reference
"""
from __future__ import annotations

import numpy as np
import pytest

from adctoolbox import fractional_delay_farrow, fractional_delay_fft


# -------------------- fractional_delay_fft --------------------


def test_fft_delay_zero_is_identity():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(512)
    y = fractional_delay_fft(x, delay_sec=0.0, fs=1.0)
    np.testing.assert_allclose(y, x, atol=1e-12)


def test_fft_delay_one_sample_matches_roll():
    # Periodic, bandlimited signal -> one-sample FFT delay equals np.roll
    N = 256
    n = np.arange(N)
    x = np.sin(2 * np.pi * 5 * n / N) + 0.3 * np.cos(2 * np.pi * 17 * n / N)
    y = fractional_delay_fft(x, delay_sec=1.0, fs=1.0)
    np.testing.assert_allclose(y, np.roll(x, 1), atol=1e-10)


def test_fft_delay_half_sample_tone():
    # Half-sample shift of a cosine is analytic
    N = 128
    k = 8
    n = np.arange(N)
    x = np.cos(2 * np.pi * k * n / N)
    y = fractional_delay_fft(x, delay_sec=0.5, fs=1.0)
    expected = np.cos(2 * np.pi * k * (n - 0.5) / N)
    np.testing.assert_allclose(y, expected, atol=1e-10)


# -------------------- fractional_delay_farrow --------------------


def test_farrow_zero_delay_is_identity_in_interior():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(256)
    y = fractional_delay_farrow(x, delay_sec=0.0, fs=1.0, n_taps=7)
    # ignore the (n_taps // 2)-sample boundary transient on each side
    np.testing.assert_allclose(y[3:-3], x[3:-3], atol=1e-12)


def test_farrow_integer_delay():
    N = 256
    n = np.arange(N)
    x = np.sin(2 * np.pi * 5 * n / N)
    y = fractional_delay_farrow(x, delay_sec=2.0, fs=1.0, n_taps=7)
    expected = np.zeros(N)
    expected[2:] = x[:-2]
    np.testing.assert_allclose(y[5:-5], expected[5:-5], atol=1e-10)


def test_farrow_half_sample_tone_reasonable_accuracy():
    N = 1024
    k = 8  # well below Nyquist -> Lagrange truncation error is tiny
    n = np.arange(N)
    x = np.cos(2 * np.pi * k * n / N)
    y = fractional_delay_farrow(x, delay_sec=0.5, fs=1.0, n_taps=7)
    expected = np.cos(2 * np.pi * k * (n - 0.5) / N)
    err = y[10:-10] - expected[10:-10]
    err_db = 20 * np.log10(max(np.abs(err).max(), 1e-300) / 1.0)
    assert err_db < -60, f"7-tap Lagrange on low-freq tone too noisy: {err_db:.1f} dB"


def test_farrow_rejects_even_taps():
    with pytest.raises(ValueError):
        fractional_delay_farrow(np.zeros(32), 0.1, 1.0, n_taps=6)
