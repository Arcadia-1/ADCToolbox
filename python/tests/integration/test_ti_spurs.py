"""Integration test: predict_spurs output matches analyze_spectrum measurement.

Flow:
    1. Synthesize a TI-ADC tone with known gain / offset / skew mismatch
    2. Run extract_mismatch_sine to recover the params
    3. Run predict_spurs to get expected spur frequencies + dBc
    4. Run analyze_spectrum on the same signal, read spur magnitudes at the
       predicted bins, and confirm they match within ~1 dB

If this test passes, the first-order TI-mismatch model is consistent with what
a user would see in a real capture.
"""
from __future__ import annotations

import numpy as np
import pytest

from adctoolbox import (
    extract_mismatch_sine,
    predict_spurs,
    analyze_spectrum,
    freq_to_bin,
)


def _synthesize(M, N, fs, fin, A, gain, offset, skew):
    T = 1.0 / fs
    x = np.empty(N)
    for n in range(N):
        m = n % M
        t = n * T + skew[m]
        x[n] = gain[m] * A * np.cos(2 * np.pi * fin * t) + offset[m]
    return x


@pytest.mark.parametrize("M", [2, 4])
def test_predict_matches_measurement(M):
    fs = 1e9
    N = 8192
    fin = 37 * fs / N            # prime-ish integer bin for coherent sampling
    A = 0.5

    rng = np.random.default_rng(42)
    gain = 1.0 + 0.02 * rng.standard_normal(M)
    offset = 0.005 * rng.standard_normal(M)
    skew = 2e-12 * rng.standard_normal(M)
    skew -= skew.mean()

    x = _synthesize(M, N, fs, fin, A, gain, offset, skew)

    params = extract_mismatch_sine(x, M, fs=fs, fin=fin)
    predicted = predict_spurs(params, fs=fs, full_scale=A)

    # One-sided peak-amplitude spectrum. The DC bin (k=0) and Nyquist bin (k=N/2)
    # are their own mirror images, so they do NOT get the 2x one-sided factor.
    spec = np.abs(np.fft.rfft(x)) / N
    spec[1:-1] *= 2.0
    spec_freqs = np.fft.rfftfreq(N, d=1.0 / fs)

    # Ignore the fundamental when comparing spurs
    fund_bin = int(round(fin / (fs / N)))

    # The two largest predicted spurs should also be visible as peaks in spec
    predicted_nontrivial = sorted(
        [s for s in predicted if s["amp"] > 1e-6],
        key=lambda s: -s["amp"],
    )
    assert predicted_nontrivial, "no predicted spurs above threshold — seed bad?"

    for s in predicted_nontrivial[:3]:   # check top few spurs
        f_pred = s["freq_hz"]
        # find nearest bin; allow +/-1 bin for rounding
        k = int(round(f_pred / (fs / N)))
        if k == fund_bin or k <= 0 or k >= len(spec):
            continue   # skip if it lands on the fundamental (expected for low-M cases)
        measured_amp = spec[max(k-1, 0): k+2].max()   # peak in ±1 bin neighborhood
        # Expect agreement within 1 dB for a first-order model + coherent FFT
        ratio_db = 20 * np.log10(measured_amp / s["amp"])
        assert abs(ratio_db) < 1.0, (
            f"spur mismatch at f={f_pred:.3g} Hz ({s['kind']}): "
            f"predicted amp={s['amp']:.3e}, measured={measured_amp:.3e}, "
            f"delta={ratio_db:+.2f} dB"
        )
