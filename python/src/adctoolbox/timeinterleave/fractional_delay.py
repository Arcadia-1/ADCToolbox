"""Fractional-sample delay primitives.

Two equivalent interfaces for applying a sub-sample time shift to a uniformly
sampled signal — used by TI-ADC skew calibration but independently useful
anywhere a fractional delay is needed.

- :func:`fractional_delay_fft`    — frequency-domain phase rotation
- :func:`fractional_delay_farrow` — time-domain Lagrange FIR (causal, streaming-friendly)

Both obey the same sign convention: ``delay_sec > 0`` shifts the signal **later**
in time (``y(t) = x(t - delay_sec)``); ``delay_sec < 0`` advances it.
"""
from __future__ import annotations

import numpy as np


def fractional_delay_fft(x: np.ndarray, delay_sec: float, fs: float) -> np.ndarray:
    """
    Apply a fractional-sample delay via DFT phase rotation.

    Output ``y[n]`` approximates ``x((n / fs) - delay_sec)`` using the sinc
    interpolation implicit in a length-``N`` DFT. Exact for signals that are
    periodic over their length and strictly bandlimited to ``[0, fs/2)``.

    Parameters
    ----------
    x : array_like, 1-D real
    delay_sec : float
        Delay in seconds; positive = later in time, negative = earlier.
    fs : float
        Sample rate (Hz).

    Returns
    -------
    y : ndarray, same length as ``x``.

    Notes
    -----
    Because the DFT assumes periodic extension, non-periodic signals will wrap
    around when ``|delay_sec|`` approaches the full record length. For clean
    results near the edges, zero-pad ``x`` before calling and trim afterwards.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"expected 1-D input, got shape {x.shape}")
    N = x.size
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(N, d=1.0 / fs)
    X = X * np.exp(-1j * 2 * np.pi * f * delay_sec)
    if N % 2 == 0:
        # Nyquist bin must stay real to preserve real-valued irfft output
        X[-1] = X[-1].real
    return np.fft.irfft(X, n=N)


def fractional_delay_farrow(
    x: np.ndarray,
    delay_sec: float,
    fs: float,
    n_taps: int = 7,
) -> np.ndarray:
    """
    Apply a fractional-sample delay via a Lagrange FIR interpolator.

    Causal / streaming alternative to :func:`fractional_delay_fft`. The
    ``n_taps``-tap centered filter trades accuracy against boundary transient
    length (``n_taps // 2`` samples on each end are unreliable, because the
    `same`-mode convolution zero-pads the input edges).

    Parameters
    ----------
    x : array_like, 1-D real
    delay_sec : float
    fs : float
    n_taps : int, default 7
        Must be a positive odd integer. 5–9 is typical; higher = flatter
        passband but longer transient.

    Returns
    -------
    y : ndarray, same length as ``x``.

    Notes
    -----
    The delay is split into an integer part (applied by zero-padded shift) and
    a fractional remainder in ``(-0.5, 0.5]`` handled by the centered Lagrange
    filter. When ``delay_sec == 0`` the filter is an impulse at the center tap.
    """
    if n_taps % 2 == 0 or n_taps < 1:
        raise ValueError(f"n_taps must be a positive odd integer, got {n_taps}")
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"expected 1-D input, got shape {x.shape}")

    D = delay_sec * fs
    D_int = int(np.round(D))
    d_frac = D - D_int

    y = _integer_shift_zero_pad(x, D_int)

    # Centered Lagrange taps: evaluation point p = P + d_frac within [0, n_taps-1].
    # d_frac == 0 yields a delta at k = P (identity).
    P = n_taps // 2
    p_eval = P + d_frac
    h = np.ones(n_taps)
    for i in range(n_taps):
        for j in range(n_taps):
            if j != i:
                h[i] *= (p_eval - j) / (i - j)
    return np.convolve(y, h, mode="same")


def _integer_shift_zero_pad(x: np.ndarray, D: int) -> np.ndarray:
    """Shift ``x`` by ``D`` samples with zero-padded boundaries (no wrap-around)."""
    if D == 0:
        return x.copy()
    y = np.zeros_like(x)
    if D > 0:
        y[D:] = x[:-D]
    else:
        y[:D] = x[-D:]
    return y
