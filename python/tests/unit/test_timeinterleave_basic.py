"""Unit tests for the time-interleave primitive helpers.

Focus: pure-numeric sanity checks that don't need plotting / file I/O —
- deinterleave / interleave round-trip
- shape / error-path coverage
- extract_mismatch_sine on a synthetic M-channel signal with known parameters
- predict_spurs produces the right number of spurs at the right bins
"""
from __future__ import annotations

import numpy as np
import pytest

from adctoolbox import (
    deinterleave,
    interleave,
    extract_mismatch_sine,
    predict_spurs,
)


# -------------------- deinterleave / interleave --------------------


def test_deinterleave_round_trip():
    x = np.arange(32.0)
    ch = deinterleave(x, 4)
    assert ch.shape == (4, 8)
    # channel m should contain x[m], x[m+4], x[m+8], ...
    for m in range(4):
        np.testing.assert_array_equal(ch[m], x[m::4])
    # and re-interleave is an inverse
    np.testing.assert_array_equal(interleave(ch), x)


def test_deinterleave_rejects_misaligned_length():
    with pytest.raises(ValueError):
        deinterleave(np.arange(10.0), 4)   # 10 not divisible by 4


def test_deinterleave_rejects_wrong_dim():
    with pytest.raises(ValueError):
        deinterleave(np.zeros((4, 4)), 2)


# -------------------- extract_mismatch_sine --------------------


def _make_ti_signal(
    M: int,
    N: int,
    fs: float,
    fin: float,
    A: float,
    gain: np.ndarray,
    offset: np.ndarray,
    skew: np.ndarray,
) -> np.ndarray:
    """Synthesize an interleaved tone with known per-channel mismatches."""
    T = 1.0 / fs
    x = np.empty(N)
    for n in range(N):
        m = n % M
        t = n * T + skew[m]     # actual sampling time seen by channel m
        x[n] = gain[m] * A * np.cos(2 * np.pi * fin * t) + offset[m]
    return x


def test_extract_mismatch_recovers_known_parameters():
    M = 4
    fs = 1e9
    # Coherent tone: choose fin so fin * N / fs is an odd integer << N
    N = 4096
    fin = 7 * fs / N          # integer bin, guaranteed coherent
    A = 0.8

    rng = np.random.default_rng(0)
    true_gain = np.array([1.00, 1.02, 0.98, 1.01])
    true_offset = np.array([0.0, 0.005, -0.003, 0.002])
    true_skew = np.array([0.0, 2e-12, -1e-12, 0.5e-12])  # ps-level
    # Recenter skew to mean zero — that's what the extractor returns
    true_skew -= true_skew.mean()

    x = _make_ti_signal(M, N, fs, fin, A, true_gain, true_offset, true_skew)
    out = extract_mismatch_sine(x, M, fs=fs, fin=fin)

    # Amplitude and normalized gain recovery
    assert out["fin"] == pytest.approx(fin)
    assert out["A"] == pytest.approx(A * true_gain.mean(), rel=1e-3)

    normalized_true_gain = true_gain / true_gain.mean()
    np.testing.assert_allclose(out["gain"], normalized_true_gain, atol=1e-4)
    np.testing.assert_allclose(out["offset"], true_offset, atol=1e-4)
    # skew should be recovered to well within a picosecond
    np.testing.assert_allclose(out["skew"], true_skew, atol=1e-13)


# -------------------- predict_spurs --------------------


def test_predict_spurs_count_and_bins():
    # 4-channel, only offset mismatch
    params = {
        "fin": 1e6,
        "A": 1.0,
        "gain": np.ones(4),
        "offset": np.array([0.0, 0.01, -0.005, 0.003]),
        "skew": np.zeros(4),
    }
    fs = 1e9
    spurs = predict_spurs(params, fs=fs, full_scale=1.0)

    # (M-1) offset + (M-1) gain_skew = 2*(M-1) = 6
    assert len(spurs) == 2 * 3
    offset_spurs = [s for s in spurs if s["kind"] == "offset"]
    gs_spurs = [s for s in spurs if s["kind"] == "gain_skew"]
    assert len(offset_spurs) == 3
    assert len(gs_spurs) == 3

    # With gain=1, skew=0 -> every gain_skew spur amplitude is 0 (-inf dB)
    for s in gs_spurs:
        assert s["amp"] == pytest.approx(0.0, abs=1e-15)

    # Offset spur frequencies land at fs/4, fs/2, (and fs·3/4 folded -> fs/4)
    freqs = sorted(s["freq_hz"] for s in offset_spurs)
    expected = sorted([fs/4, fs/2, fs/4])
    np.testing.assert_allclose(freqs, expected, rtol=1e-12)
