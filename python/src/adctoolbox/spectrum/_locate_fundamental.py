"""Fundamental frequency location with sub-bin accuracy.

This module provides the _locate_fundamental() function for finding the
fundamental frequency bin in a power spectrum with parabolic interpolation.

This is an internal helper module, not intended for direct use by end users.
"""

import numpy as np


def _locate_fundamental(
    spectrum: np.ndarray,
    n_inband: int
) -> tuple[int, float]:
    """Locate the fundamental frequency bin in power spectrum.

    Finds the peak in the in-band region and applies parabolic interpolation
    for sub-bin accuracy.

    Parameters
    ----------
    spectrum : np.ndarray
        Power spectrum (|FFT|^2)
    n_inband : int
        In-band search limit (typically N//2//osr)

    Returns
    -------
    fundamental_bin : int
        Integer index of the fundamental bin (0-based)
    fundamental_bin_frantional : float
        Refined bin location with parabolic interpolation

    Notes
    -----
    - Searches only in-band region [1, n_inband)
    - Excludes DC bin (index 0) from search
    - Uses parabolic interpolation for sub-bin accuracy
    - Returns both integer and refined bin positions
    """
    # Search in-band region only
    spectrum_search = spectrum[:n_inband]

    # Find peak (excluding DC bin at index 0)
    # Start from bin 1 to avoid DC
    if len(spectrum_search) > 1:
        fundamental_bin = np.argmax(spectrum_search[1:]) + 1
    else:
        fundamental_bin = 0

    # Parabolic interpolation for refined bin location
    # Using 3 points: (bin-1, bin, bin+1)
    fundamental_bin_frantional = float(fundamental_bin)

    # Get the three points for interpolation
    if fundamental_bin > 0 and fundamental_bin < len(spectrum_search) - 1:
        # Convert to log scale for interpolation (more robust)
        y_m1 = np.log10(max(spectrum_search[fundamental_bin - 1], 1e-20))
        y_0 = np.log10(max(spectrum_search[fundamental_bin], 1e-20))
        y_p1 = np.log10(max(spectrum_search[fundamental_bin + 1], 1e-20))

        # Parabolic interpolation formula
        delta = (y_p1 - y_m1) / (2 * (2 * y_0 - y_m1 - y_p1))
        fundamental_bin_frantional = fundamental_bin + delta

        # Check for invalid result
        if np.isnan(fundamental_bin_frantional) or np.isinf(fundamental_bin_frantional):
            fundamental_bin_frantional = float(fundamental_bin)

    return fundamental_bin, fundamental_bin_frantional
