"""Find fundamental frequency in FFT spectrum - shared helper.

This module provides function to locate the fundamental frequency bin
in an FFT spectrum, including parabolic interpolation for sub-bin accuracy.

This is an internal helper module, not intended for direct use by end users.
"""

import numpy as np
from typing import Tuple


def _find_fundamental(
    spectrum: np.ndarray,
    n_fft: int,
    osr: int = 1,
    method: str = 'magnitude'
) -> Tuple[int, float]:
    """Find the fundamental frequency bin in FFT spectrum.

    Locates the fundamental frequency using the specified method and applies
    parabolic interpolation for sub-bin accuracy.

    Parameters
    ----------
    spectrum : np.ndarray
        FFT spectrum data (complex or magnitude squared)
    n_fft : int
        FFT length
    osr : int, optional
        Oversampling ratio, default is 1
    method : str, optional
        Method for finding fundamental:
        - 'magnitude': Use magnitude spectrum (default)
        - 'power': Use power spectrum (|spectrum|^2)
        - 'log': Use logarithmic magnitude for better dynamic range

    Returns
    -------
    bin_idx : int
        Integer index of the fundamental bin (0-based)
    bin_r : float
        Refined bin location with parabolic interpolation

    Notes
    -----
    - Searches only in-band (up to Nyquist/OSR)
    - Uses parabolic interpolation for sub-bin accuracy
    - Returns both integer and refined bin positions
    """
    # Determine search range (in-band only)
    n_search = n_fft // 2 // osr
    spectrum_search = spectrum[:n_search]

    # Convert to appropriate metric for search
    if method == 'magnitude':
        search_metric = np.abs(spectrum_search)
    elif method == 'power':
        search_metric = np.abs(spectrum_search)**2
    elif method == 'log':
        search_metric = 20 * np.log10(np.abs(spectrum_search) + 1e-20)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Find peak (excluding DC bin at index 0)
    # Start from bin 1 to avoid DC
    if len(search_metric) > 1:
        bin_idx = np.argmax(search_metric[1:]) + 1
    else:
        bin_idx = 0

    # Parabolic interpolation for refined bin location
    # Using 3 points: (bin-1, bin, bin+1)
    bin_r = float(bin_idx)

    # Get the three points for interpolation
    if bin_idx > 0 and bin_idx < len(search_metric) - 1:
        # Convert to log scale for interpolation (more robust)
        y_m1 = np.log10(max(search_metric[bin_idx - 1], 1e-20))
        y_0 = np.log10(max(search_metric[bin_idx], 1e-20))
        y_p1 = np.log10(max(search_metric[bin_idx + 1], 1e-20))

        # Parabolic interpolation formula
        delta = (y_p1 - y_m1) / (2 * (2 * y_0 - y_m1 - y_p1))
        bin_r = bin_idx + delta

        # Check for invalid result
        if np.isnan(bin_r) or np.isinf(bin_r):
            bin_r = float(bin_idx)

    return bin_idx, bin_r
