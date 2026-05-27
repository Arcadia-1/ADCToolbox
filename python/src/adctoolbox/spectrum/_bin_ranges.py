"""FFT bin range helpers shared by spectrum analysis code."""

import numpy as np


def rfft_inband_bin_count(n_fft: int, osr: float = 1) -> int:
    """Return the number of rFFT bins inside the analysis bandwidth.

    The returned value is intended for Python slices such as
    ``spectrum[:n_bins]``. It includes DC and the upper analysis-band edge.

    For ``osr=1`` this means:
    - even ``N``: include bins ``0..N/2`` including Nyquist
    - odd ``N``: include bins ``0..floor(N/2)``; there is no Nyquist bin
    """
    if n_fft <= 0:
        raise ValueError("n_fft must be positive")
    if osr <= 0:
        raise ValueError("osr must be positive")

    rfft_len = n_fft // 2 + 1
    edge_bin = n_fft / (2 * osr)
    nearest_int = np.rint(edge_bin)
    if np.isclose(edge_bin, nearest_int, rtol=1e-12, atol=1e-12):
        edge_bin = nearest_int
    count = int(np.floor(edge_bin)) + 1
    return max(1, min(count, rfft_len))
