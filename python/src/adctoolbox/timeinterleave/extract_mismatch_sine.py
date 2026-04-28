"""Per-channel mismatch extraction for a time-interleaved ADC.

Given an interleaved single-tone output ``x``, de-interleave into ``M`` sub-channels,
fit a tone at ``fin`` to each channel's samples (via DFT at the known frequency),
and return offset / gain / sample-skew per channel.

The complex-phasor method used here is equivalent to a 4-parameter sine fit when
the signal is coherent and the tone frequency is known.
"""
from __future__ import annotations

import numpy as np

from adctoolbox.timeinterleave.deinterleave import deinterleave
from adctoolbox.fundamentals.frequency import estimate_frequency


def extract_mismatch_sine(
    x: np.ndarray,
    M: int,
    fs: float,
    fin: float | None = None,
) -> dict:
    """
    Extract per-channel offset, gain, and sample-skew from a TI-ADC sine capture.

    Parameters
    ----------
    x : array_like, shape (N,)
        Interleaved output. Sample ``x[n]`` belongs to channel ``n mod M`` and was
        taken at time ``n / fs``.
    M : int
        Number of sub-ADCs.
    fs : float
        Aggregate sample rate (Hz).
    fin : float, optional
        Input-tone frequency. If None, it is estimated from the FFT of ``x``
        (use coherent sampling for best results).

    Returns
    -------
    params : dict
        ``gain``    : (M,) relative gain, normalized so ``mean == 1``.
        ``offset``  : (M,) DC offset per channel (same units as ``x``).
        ``skew``    : (M,) sample-skew per channel (seconds, mean zero).
        ``fin``     : float — tone frequency used for the fit.
        ``A``       : float — fitted fundamental amplitude (mean across channels).
        ``phases``  : (M,) raw fitted phase per channel (rad) for diagnostics.

    Notes
    -----
    The "skew" returned here is relative: the mean is subtracted so an overall
    clock delay (which is not observable from a single capture) does not leak
    into the per-channel result. Pass ``skew - skew.mean()`` upstream.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"expected 1-D input, got shape {x.shape}")
    if M <= 0:
        raise ValueError(f"M must be positive, got {M}")
    N = x.size
    if N % M != 0:
        raise ValueError(f"len(x)={N} is not a multiple of M={M}")

    if fin is None:
        fin = estimate_frequency(x, fs=fs)

    channels = deinterleave(x, M)           # (M, K), K = N / M
    K = channels.shape[1]
    T = 1.0 / fs

    # Time of sample k in channel m: t = (k*M + m) * T
    # Phasor at fin: P_m = (2/K) * Σ (y - offset) * exp(-j 2π fin t)
    phasors = np.empty(M, dtype=complex)
    offsets = np.empty(M, dtype=float)
    for m in range(M):
        y = channels[m]
        offsets[m] = y.mean()
        k = np.arange(K)
        t = (k * M + m) * T
        phasors[m] = (2.0 / K) * np.sum((y - offsets[m]) * np.exp(-1j * 2 * np.pi * fin * t))

    amps = np.abs(phasors)
    phases = np.angle(phasors)          # phase already referenced to absolute t=0

    A_mean = amps.mean()
    gain = amps / A_mean if A_mean > 0 else np.ones(M)

    # Residual phase after accounting for the m*T interleave delay
    # (that delay is baked into the exponential above, so any non-zero residual
    # phase spread is the skew signature).
    # Unwrap across channels first, then subtract mean (unobservable overall delay).
    phases_u = np.unwrap(phases)
    skew = (phases_u - phases_u.mean()) / (2 * np.pi * fin)

    return {
        "fin": float(fin),
        "A": float(A_mean),
        "gain": gain,
        "offset": offsets,
        "skew": skew,
        "phases": phases,
    }
