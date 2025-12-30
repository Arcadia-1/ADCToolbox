"""Helper function to exclude bins from spectrum for noise calculation."""

import numpy as np

def _exclude_bins_from_spectrum(
    spectrum: np.ndarray,
    signal_bin: int,
    harmonic_bins: np.ndarray,
    side_bin: int,
    max_bin: int
) -> np.ndarray:
    """Exclude DC, signal, and harmonic bins from spectrum for noise calculation.

    This function creates a copy of the spectrum and zeros out bins corresponding
    to DC, the fundamental signal, and all harmonics (including their side bins).

    Parameters
    ----------
    spectrum : np.ndarray
        Input power spectrum
    signal_bin : int
        Fundamental signal bin index
    harmonic_bins : np.ndarray
        Array of harmonic bin positions [H2, H3, H4, ...] (does NOT include fundamental)
    side_bin : int
        Number of side bins to exclude around each component
    max_bin : int
        Maximum bin index to consider (typically n_fft // 2 // osr)

    Returns
    -------
    np.ndarray
        Spectrum with excluded bins set to zero
    """
    # Create a copy to avoid modifying the original
    noise_spectrum = spectrum[:max_bin].copy()

    # Exclude DC and side bins
    noise_spectrum[0:side_bin+1] = 0

    # Exclude fundamental signal
    if 1 <= signal_bin < len(noise_spectrum):
        sig_start = max(signal_bin - side_bin, 0)
        sig_end = min(signal_bin + side_bin + 1, len(noise_spectrum))
        noise_spectrum[sig_start:sig_end] = 0

    # Exclude all harmonics (H2, H3, H4, ...)
    # harmonic_bins contains only harmonics [H2, H3, H4, ...], not the fundamental
    for h_idx in range(len(harmonic_bins)):
        h_bin = harmonic_bins[h_idx]  # Already int from _locate_harmonic_bins
        if 1 <= h_bin < len(noise_spectrum):
            h_start = max(h_bin - side_bin, 0)
            h_end = min(h_bin + side_bin + 1, len(noise_spectrum))
            noise_spectrum[h_start:h_end] = 0

    return noise_spectrum
