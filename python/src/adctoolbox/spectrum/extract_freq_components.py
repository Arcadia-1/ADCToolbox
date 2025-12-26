# Not yet verified with both MATLAB and Python testbenches
import numpy as np
from ..fundamentals.frequency import fold_frequency_to_nyquist

def extract_freq_components(din, bands):
    """
    Extract signal components within specified frequency bands.

    Args:
        din: Input data matrix (N x M or M x N)
        bands: Frequency bands matrix (P x 2), each row contains [low_freq, high_freq]

    Returns:
        dout: Output data with only components in specified bands
    """
    N, M = din.shape
    if N < M:
        din = din.T
        N, M = din.shape

    P, Q = bands.shape
    if Q != 2:
        raise ValueError('bands must be with 2 columns')

    spec = np.fft.fft(din, axis=0)
    mask = np.zeros((N, 1))

    for i1 in range(P):
        n1 = int(round(min(bands[i1, :]) * N))
        n2 = int(round(max(bands[i1, :]) * N))

        # Generate frequency indices with aliasing
        freq_indices = np.arange(n1, n2 + 1)
        ids = np.array([fold_frequency_to_nyquist(j, N) for j in freq_indices])

        # Set mask for positive frequencies
        mask[ids + 1, 0] = 1
        # Set mask for negative frequencies (mirror)
        mask[N - ids - 1, 0] = 1

    # Apply mask to spectrum
    spec = spec * (mask @ np.ones((1, M)))
    dout = np.real(np.fft.ifft(spec, axis=0))

    return dout
