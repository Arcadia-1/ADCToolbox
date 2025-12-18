"""Find harmonic bin positions with aliasing - shared helper.

This module provides function to calculate harmonic bin positions for a given
fundamental frequency, properly handling aliasing for real signals.

This is an internal helper module, not intended for direct use by end users.
"""

import numpy as np
from adctoolbox.fundamentals.frequency import fold_frequency_to_nyquist


def _find_harmonic_bins(
    fundamental_bin: float,
    harmonic: int,
    n_fft: int
) -> np.ndarray:
    """Find harmonic bin positions with aliasing.

    Calculates the bin positions for harmonics of the fundamental frequency,
    handling aliasing for real signals using fold_frequency_to_nyquist().

    Parameters
    ----------
    fundamental_bin : float
        Fundamental bin position (can be fractional from interpolation)
    harmonic : int
        Number of harmonics to find
    n_fft : int
        FFT length

    Returns
    -------
    harmonic_bins : np.ndarray
        Array of harmonic bin positions (0-based)

    Notes
    -----
    - Uses fold_frequency_to_nyquist() from common module for aliasing calculation
    - For real signals, frequencies above Nyquist are aliased/mirrored
    - Harmonic h appears at frequency = h * fundamental_frequency
    - Bin index = h * fundamental_bin (mod n_fft)
    """
    harmonic_bins = np.zeros(harmonic)

    # Convert bin to normalized frequency
    # bin / n_fft = normalized frequency (0 to 0.5 for real signals)
    # Multiply by 2 to get frequency relative to Nyquist
    fundamental_freq_normalized = fundamental_bin / n_fft

    for h in range(1, harmonic + 1):
        # Calculate harmonic frequency (normalized)
        harmonic_freq_normalized = fundamental_freq_normalized * h

        # Use calc_aliased_freq to handle aliasing
        # fs = 1.0 (normalized), so Nyquist = 0.5
        # The function returns frequency in [0, fs/2]
        aliased_freq = fold_frequency_to_nyquist(harmonic_freq_normalized, 1.0)

        # Convert back to bin index
        harmonic_bin = aliased_freq * n_fft

        # Ensure we're in the valid range
        harmonic_bin = max(0, min(harmonic_bin, n_fft // 2))

        harmonic_bins[h - 1] = harmonic_bin

    return harmonic_bins
