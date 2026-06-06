"""MATLAB-compatible ideal FFT brickwall filter."""

from __future__ import annotations

import numpy as np

from adctoolbox.spectrum.extract_freq_components import extract_freq_components


def ifilter(sigin: np.ndarray, passband: np.ndarray) -> np.ndarray:
    """Filter a signal by retaining only the specified normalized passbands.

    This is the Python counterpart of MATLAB ``ifilter(sigin, passband)``.
    It applies an ideal FFT-domain brickwall mask, preserves Hermitian
    symmetry for real-valued output, and filters each column independently.

    Parameters
    ----------
    sigin : array_like
        Real-valued input signal. Vectors are treated as column vectors; for
        matrices, each column is filtered independently. If the matrix is
        wider than tall, it is transposed to match the MATLAB orientation rule.
    passband : array_like
        ``(P, 2)`` passband table. Each row is ``[f_low, f_high]`` normalized
        to the sampling frequency, where ``0`` is DC and ``0.5`` is Nyquist.

    Returns
    -------
    ndarray
        Filtered signal in MATLAB-style orientation.
    """

    return extract_freq_components(sigin, passband)

