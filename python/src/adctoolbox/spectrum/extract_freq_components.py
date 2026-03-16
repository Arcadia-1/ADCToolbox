"""Ideal brickwall FFT filter for extracting frequency band components.

Applies FFT-based brickwall filters to extract specified frequency bands
from input signals. Each column is filtered independently.
"""

import numpy as np
from ..fundamentals.frequency import fold_frequency_to_nyquist


def extract_freq_components(din, bands):
    """
    Extract signal components within specified frequency bands.

    Parameters
    ----------
    din : np.ndarray
        Input data matrix (N x M or M x N). Must be real-valued.
    bands : np.ndarray
        Frequency bands matrix (P x 2), each row [low_freq, high_freq].
        Frequencies normalized: 0 = DC, 0.5 = Nyquist.

    Returns
    -------
    np.ndarray
        Output data with only components in specified bands.
    """
    din = np.asarray(din, dtype=float)
    bands = np.asarray(bands, dtype=float)

    if din.ndim == 1:
        din = din[:, np.newaxis]

    N, M = din.shape
    if N < M:
        din = din.T
        N, M = din.shape

    if bands.ndim == 1:
        bands = bands.reshape(1, -1)

    P, Q = bands.shape
    if Q != 2:
        raise ValueError('bands must have exactly 2 columns [fLow, fHigh]')

    spec = np.fft.fft(din, axis=0)
    mask = np.zeros(N)

    for i in range(P):
        n1 = int(round(min(bands[i, :]) * N))
        n2 = int(round(max(bands[i, :]) * N))

        freq_indices = np.arange(n1, n2 + 1)
        ids = np.round(fold_frequency_to_nyquist(freq_indices, N)).astype(int)

        # Set positive frequency bins
        mask[ids] = 1

        # Set negative frequency bins (Hermitian symmetry)
        # Exclude DC (0) and Nyquist (N/2) to avoid double-setting
        valid = ids[(ids > 0) & (ids < N // 2)]
        mask[N - valid] = 1

    # Apply mask to all columns
    spec = spec * mask[:, np.newaxis]
    dout = np.real(np.fft.ifft(spec, axis=0))

    return dout
