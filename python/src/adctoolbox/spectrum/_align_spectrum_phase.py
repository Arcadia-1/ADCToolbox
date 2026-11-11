"""
Align spectrum phase for coherent averaging.

This module implements the phase alignment algorithm for FFT coherent mode.
"""

import numpy as np

def _align_spectrum_phase(
    fft_data: np.ndarray,
    bin_idx: int,
    bin_r: float,
    n_fft: int,
    lobe_half_width: int = 0,
) -> np.ndarray:
    """
    Align the phase of an FFT spectrum to phase 0 at the fundamental frequency.

    This function performs phase rotation to align harmonics and non-harmonics
    separately using a two-step process:
    1. Harmonic alignment with Nyquist folding detection
    2. Non-harmonic alignment with fractional phase

    Parameters
    ----------
    fft_data : np.ndarray
        Complex FFT data to align
    bin_idx : int
        Fundamental frequency bin index
    bin_r : float
        Refined fundamental frequency bin (with sub-bin precision)
    n_fft : int
        FFT length
    lobe_half_width : int, default=0
        Number of bins on each side of a harmonic that belong to its window
        main lobe (0 for a rectangular window, 1 for Hann, ...). Those bins
        carry the harmonic's phase and are rotated with it.

    Returns
    -------
    np.ndarray
        Phase-aligned FFT spectrum (complex)

    Notes
    -----
    Algorithm:
    - Harmonic h is rotated by h times the fundamental phase, together with
      the ``lobe_half_width`` bins on each side of it. A window spreads a
      coherent line over those bins with the line's own phase, so giving them
      any other rotation makes the lobe average away across runs.
    - Nyquist zone detection: even zones use normal phase, odd zones use conjugate
    - When several harmonics claim a bin, the lowest order wins
    - Non-harmonics are rotated by fractional phase based on bin position
    - DC bin is zeroed after alignment
    """
    # Guard against DC bin
    if bin_idx <= 0:
        return fft_data

    # Calculate phase rotation to align fundamental to phase 0
    fundamental_phasor = fft_data[bin_idx] / (np.abs(fft_data[bin_idx]) + 1e-20)
    rotation_angle = -np.angle(fundamental_phasor)

    n_half = n_fft // 2
    lobe_half_width = int(max(lobe_half_width, 0))

    # Step 1: Align harmonics with Nyquist folding.
    # Harmonic h sits at (h * bin_idx) mod n_fft. In an even Nyquist zone that
    # position is in the lower half of the spectrum and rotates by h times the
    # fundamental rotation; in an odd zone its mirror image is, and rotates by
    # the conjugate.
    orders = np.arange(1, n_fft + 1, dtype=np.int64)
    folded = (orders * int(bin_idx)) % n_fft
    odd_zone = 2 * folded >= n_fft
    centers = np.where(odd_zone, n_fft - folded, folded)

    # Lowest harmonic order whose lobe covers each lower-half bin.
    no_owner = n_fft + 1
    owner = np.full(n_half + 1, no_owner, dtype=np.int64)
    for offset in range(-lobe_half_width, lobe_half_width + 1):
        positions = centers + offset
        inside = (positions >= 0) & (positions <= n_half)
        np.minimum.at(owner, positions[inside], orders[inside])

    fft_aligned = fft_data.copy()
    owned = owner != no_owner
    owner_orders = owner[owned]
    signed_orders = np.where(odd_zone[owner_orders - 1], -owner_orders, owner_orders)
    lower_half = fft_aligned[: n_half + 1]
    lower_half[owned] *= np.exp(1j * rotation_angle * signed_orders)

    # Step 2: Align non-harmonics with fractional phase based on bin position
    # relative to the refined fundamental.
    if bin_r > 0:
        processed = np.zeros(n_fft, dtype=bool)
        processed[: n_half + 1] = owned
        remaining = np.flatnonzero(~processed)
        fft_aligned[remaining] *= np.exp(1j * rotation_angle * remaining / bin_r)

    # Zero out DC component
    fft_aligned[0] = 0

    return fft_aligned
