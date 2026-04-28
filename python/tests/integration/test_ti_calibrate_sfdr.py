"""Integration test: calibrate_foreground lowers TI spurs by a clear margin.

Synthesize a TI-ADC capture with realistic offset / gain / skew mismatch, run
both FFT-based and Farrow-based calibration, and confirm each pushes the
worst TI spur below its expected floor.

Reference floors (at the parameter set used here):

- Uncalibrated SFDR:     ~45 dBc (dominated by skew-induced spurs at fin ± k·fs/M)
- FFT-calibrated SFDR:   > 100 dBc (residual == numerical precision)
- Farrow-9 SFDR:         > 80 dBc (residual == truncation of Lagrange kernel)
"""
from __future__ import annotations

import numpy as np
import pytest

from adctoolbox import (
    calibrate_foreground,
    extract_mismatch_sine,
)


def _synth(M, N, fs, fin, A, gain, offset, skew):
    T = 1.0 / fs
    x = np.empty(N)
    for n in range(N):
        m = n % M
        t = n * T + skew[m]
        x[n] = gain[m] * A * np.cos(2 * np.pi * fin * t) + offset[m]
    return x


def _sfdr_dbc(x: np.ndarray, N: int, fs: float, fin: float) -> float:
    """Crude SFDR: worst non-fundamental peak of the peak-amplitude spectrum."""
    spec = np.abs(np.fft.rfft(x)) / N
    spec[1:-1] *= 2.0
    fund_bin = int(round(fin / (fs / N)))
    fund_amp = spec[fund_bin]
    # Mask out DC, fundamental (and one neighbor bin on each side for leakage)
    mask = np.ones_like(spec, dtype=bool)
    mask[0] = False
    for b in range(max(fund_bin - 1, 0), min(fund_bin + 2, len(spec))):
        mask[b] = False
    worst = spec[mask].max()
    return 20 * np.log10(fund_amp / max(worst, 1e-300))


@pytest.mark.parametrize("skew_method,floor_db,n_taps", [
    ("fft",    100.0, 7),   # FFT method → numerical-precision floor
    ("farrow",  80.0, 9),   # 9-tap Lagrange → still >80 dBc with these sizes
])
def test_calibration_lifts_sfdr_above_floor(skew_method, floor_db, n_taps):
    M = 4
    fs = 1e9
    N = 8192
    fin = 37 * fs / N
    A = 0.5

    rng = np.random.default_rng(7)
    gain = 1.0 + 0.02 * rng.standard_normal(M)
    offset = 0.005 * rng.standard_normal(M)
    skew = 3e-12 * rng.standard_normal(M)
    skew -= skew.mean()

    x = _synth(M, N, fs, fin, A, gain, offset, skew)
    sfdr_before = _sfdr_dbc(x, N, fs, fin)

    params = extract_mismatch_sine(x, M, fs=fs, fin=fin)
    y = calibrate_foreground(x, M, params, fs, skew_method=skew_method, n_taps=n_taps)
    sfdr_after = _sfdr_dbc(y, N, fs, fin)

    assert sfdr_before < 60.0, (
        f"uncalibrated capture unexpectedly clean: SFDR={sfdr_before:.1f} dBc — "
        f"test seed likely produced trivial mismatch, pick a different one."
    )
    assert sfdr_after > floor_db, (
        f"{skew_method} calibration didn't meet floor: "
        f"before={sfdr_before:.1f} dBc, after={sfdr_after:.1f} dBc, "
        f"required >{floor_db} dBc"
    )
