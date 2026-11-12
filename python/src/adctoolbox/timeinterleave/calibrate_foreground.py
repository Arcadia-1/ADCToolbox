"""Foreground calibration of a TI-ADC capture.

Given mismatch parameters from :func:`extract_mismatch_sine`, undo the per-channel
offset, gain, and sample-skew errors and return a corrected interleaved signal.
"""
from __future__ import annotations

import warnings

import numpy as np

from adctoolbox.timeinterleave.deinterleave import deinterleave, interleave
from adctoolbox.timeinterleave.fractional_delay import (
    fractional_delay_fft,
    fractional_delay_farrow,
)


# Skew-correction passes. The FFT delay is exact for a periodic record, so the
# iteration runs until the residual reaches rounding level.
_MAX_SKEW_PASSES = 12
_FARROW_SKEW_PASSES = 3
_SKEW_RESIDUAL_TOLERANCE = 1e-14


def calibrate_foreground(
    x: np.ndarray,
    M: int,
    params: dict,
    fs: float,
    *,
    skew_method: str = "fft",
    n_taps: int = 7,
) -> np.ndarray:
    """
    Apply offset / gain / skew corrections to a TI-ADC interleaved capture.

    Parameters
    ----------
    x : array_like, shape (N,)
        Interleaved time series.
    M : int
        Sub-ADC count.
    params : dict
        Output of :func:`extract_mismatch_sine`. Required keys: ``offset``,
        ``gain``, ``skew``.
    fs : float
        Aggregate sample rate (Hz).
    skew_method : {'fft', 'farrow'}, default 'fft'
        How to apply the fractional-sample skew correction:

        - ``'fft'``   — DFT phase rotation of the full-rate stream. Near
          machine-precision accuracy for a periodic record with energy below
          ``fs/2``; circular wrap at record boundaries.
        - ``'farrow'`` — Lagrange FIR on the full-rate stream. Causal and
          streaming-friendly. Accuracy and boundary transient length are set
          by ``n_taps``; a short Lagrange filter loses accuracy towards
          ``fs/2`` (about 60 dB of SFDR at 0.48 fs with 7 taps).

    n_taps : int, default 7
        Only consulted when ``skew_method == 'farrow'``; must be a positive odd
        integer.

    Returns
    -------
    y : ndarray, shape (N,)
        Calibrated interleaved signal.

    Notes
    -----
    The order matters: offset is removed first (before gain normalization),
    then gain is applied, and skew correction is last.

    The skew is corrected on the interleaved stream at the aggregate rate
    ``fs``, not on each sub-ADC's own samples. A sub-ADC samples at ``fs/M``,
    so any input above ``fs/(2M)`` is aliased within one channel and a delay
    applied there shifts the alias, not the signal. The correction is valid
    for inputs in the first Nyquist zone of the aggregate rate; a warning is
    raised when ``params["fin"]`` lies above ``fs/2``.
    """
    x = np.asarray(x, dtype=float)
    channels = deinterleave(x, M).astype(float)
    offset = np.asarray(params["offset"], dtype=float)
    gain = np.asarray(params["gain"], dtype=float)
    skew = np.asarray(params["skew"], dtype=float)

    if not (offset.size == M and gain.size == M and skew.size == M):
        raise ValueError(
            f"params offset/gain/skew must each have length M={M}, "
            f"got {offset.size} / {gain.size} / {skew.size}"
        )

    # 1. Offset (additive) — subtract per-channel DC
    # 2. Gain (multiplicative) — divide by per-channel relative gain
    channels = (channels - offset[:, None]) / gain[:, None]

    # 3. Skew. The capture is the ideal uniformly sampled stream z resampled
    #    on a skewed grid: y[n] = z(n/fs + skew[m]) for n = m (mod M), see
    #    extract_mismatch_sine. Invert that map on the full-rate stream.
    if skew_method not in ("fft", "farrow"):
        raise ValueError(
            f"skew_method must be 'fft' or 'farrow', got {skew_method!r}"
        )
    corrected = interleave(channels)
    active = np.flatnonzero(np.abs(skew) >= 1e-18)
    if active.size == 0:
        return corrected

    fin = params.get("fin") if hasattr(params, "get") else None
    if fin is not None and float(fin) > fs / 2:
        warnings.warn(
            f"The input tone ({float(fin):g} Hz) is above fs/2 ({fs / 2:g} Hz). "
            "Skew correction interpolates the sampled stream and is only "
            "valid in the first Nyquist zone; offset and gain are still "
            "corrected.",
            UserWarning,
            stacklevel=2,
        )

    if skew_method == "fft":
        def advance(stream, seconds):
            return fractional_delay_fft(stream, -seconds, fs)
        max_passes = _MAX_SKEW_PASSES
    else:
        def advance(stream, seconds):
            return fractional_delay_farrow(stream, -seconds, fs, n_taps=n_taps)
        # The FIR's own interpolation error limits what further passes gain.
        max_passes = _FARROW_SKEW_PASSES

    def resample_on_skewed_grid(stream):
        skewed = stream.copy()
        for m in active:
            skewed[m::M] = advance(stream, skew[m])[m::M]
        return skewed

    # Fixed-point iteration z <- z + (y - A z). A differs from the identity
    # by about 2*pi*fin*skew, so each pass gains tens of dB; the first pass
    # alone is the usual first-order correction.
    estimate = corrected.copy()
    scale = np.max(np.abs(corrected))
    for _ in range(max_passes):
        residual = corrected - resample_on_skewed_grid(estimate)
        estimate += residual
        if np.max(np.abs(residual)) <= _SKEW_RESIDUAL_TOLERANCE * scale:
            break

    return estimate
