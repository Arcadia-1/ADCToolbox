"""Split an interleaved time series into M sub-channels (and back)."""
from __future__ import annotations

import numpy as np


def deinterleave(x: np.ndarray, M: int) -> np.ndarray:
    """
    Deinterleave ``x`` into ``M`` sub-channels.

    Parameters
    ----------
    x : array_like, shape (N,)
        Interleaved time series. Sample at index ``n`` belongs to channel ``n mod M``.
    M : int
        Number of sub-ADCs (channels).

    Returns
    -------
    channels : ndarray, shape (M, N // M)
        ``channels[m]`` contains the samples of channel ``m`` in chronological order.
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"expected 1-D input, got shape {x.shape}")
    if M <= 0:
        raise ValueError(f"M must be positive, got {M}")
    N = x.size
    if N % M != 0:
        raise ValueError(f"len(x)={N} is not a multiple of M={M}; truncate or pad first")
    return x.reshape(N // M, M).T


def interleave(channels: np.ndarray) -> np.ndarray:
    """
    Inverse of :func:`deinterleave` — stitch M channels back into one stream.

    Parameters
    ----------
    channels : array_like, shape (M, K)
        ``channels[m, k]`` is the k-th sample of channel m.

    Returns
    -------
    x : ndarray, shape (M * K,)
    """
    channels = np.asarray(channels)
    if channels.ndim != 2:
        raise ValueError(f"expected 2-D input (M, K), got shape {channels.shape}")
    return channels.T.reshape(-1)
