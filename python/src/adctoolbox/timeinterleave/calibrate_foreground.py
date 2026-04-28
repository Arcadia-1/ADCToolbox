"""Foreground calibration of a TI-ADC capture.

Given mismatch parameters from :func:`extract_mismatch_sine`, undo the per-channel
offset, gain, and sample-skew errors and return a corrected interleaved signal.
"""
from __future__ import annotations

import numpy as np

from adctoolbox.timeinterleave.deinterleave import deinterleave, interleave
from adctoolbox.timeinterleave.fractional_delay import (
    fractional_delay_fft,
    fractional_delay_farrow,
)


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

        - ``'fft'``   — per-channel DFT phase rotation. Near machine-precision
          accuracy for signals with energy strictly below ``fs/(2M)``; circular
          wrap at record boundaries.
        - ``'farrow'`` — per-channel Lagrange FIR. Causal and streaming-friendly.
          Accuracy and boundary transient length are set by ``n_taps``.

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
    then gain is applied, and skew correction is last — mixing channels only
    at the fractional-delay interpolation stage preserves the per-channel
    amplitude balance that the prior two steps just enforced.
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

    # 3. Skew: apply a delay of +skew[m] per channel so the calibrated sample n
    #    represents the signal at n/fs + m/fs (ideal grid). See derivation in
    #    extract_mismatch_sine: recorded y[n] = s(t_ideal + skew[m]), so
    #    z[n] = y(t - skew[m]) — i.e., y delayed by +skew[m] — recovers s(t_ideal).
    fs_ch = fs / M
    for m in range(M):
        if abs(skew[m]) < 1e-18:
            continue
        if skew_method == "fft":
            channels[m] = fractional_delay_fft(channels[m], skew[m], fs_ch)
        elif skew_method == "farrow":
            channels[m] = fractional_delay_farrow(
                channels[m], skew[m], fs_ch, n_taps=n_taps
            )
        else:
            raise ValueError(
                f"skew_method must be 'fft' or 'farrow', got {skew_method!r}"
            )

    return interleave(channels)
