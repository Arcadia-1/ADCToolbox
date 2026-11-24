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
    # Harmonic h sits at (h * bin_idx) mod n_fft. In an even Nyquist zone
    # (0 <= folded <= n_fft/2) it rotates by +h times the fundamental rotation;
    # in an odd zone (folded > n_fft/2) its mirror image rotates by -h.
    orders = np.arange(1, n_fft + 1, dtype=np.int64)
    folded = (orders * int(bin_idx)) % n_fft
    odd_zone = 2 * folded > n_fft
    centers = np.where(odd_zone, n_fft - folded, folded)

    # Assign harmonic ownership in priority passes so that:
    # 1. The fundamental (order 1) always owns its full window main lobe.
    # 2. Low-order harmonics' exact center bins are never overwritten by a
    #    lower-order harmonic's side bin (e.g. an empty H2 lobe cancelling an
    #    adjacent H3 peak).
    # 3. Side bins of low-order harmonics go to the nearest harmonic center
    #    (lowest order tie-break), then higher-order harmonic centers fill any
    #    remaining unclaimed bins.
    no_owner = n_fft + 1
    owner = np.full(n_half + 1, no_owner, dtype=np.int64)

    # Pass 1: Fundamental full lobe
    f_lo = max(0, int(centers[0]) - lobe_half_width)
    f_hi = min(n_half + 1, int(centers[0]) + lobe_half_width + 1)
    owner[f_lo:f_hi] = 1

    # Pass 2: Low-order harmonic center bins (orders 2..n_priority)
    n_priority = min(n_fft, 16)
    if n_priority >= 2:
        pri_orders = orders[1:n_priority]
        pri_centers = centers[1:n_priority]
        inside = (pri_centers >= 0) & (pri_centers <= n_half)
        pos = pri_centers[inside]
        ord_in = pri_orders[inside]
        free = owner[pos] == no_owner
        np.minimum.at(owner, pos[free], ord_in[free])

        # Pass 3: Low-order harmonic side bins by increasing distance from center
        for dist in range(1, lobe_half_width + 1):
            dist_owner = np.full(n_half + 1, no_owner, dtype=np.int64)
            for sign in (-1, 1):
                positions = pri_centers + sign * dist
                in_b = (positions >= 0) & (positions <= n_half)
                pos_b = positions[in_b]
                free_b = owner[pos_b] == no_owner
                np.minimum.at(dist_owner, pos_b[free_b], pri_orders[in_b][free_b])
            newly = (owner == no_owner) & (dist_owner != no_owner)
            owner[newly] = dist_owner[newly]

    # Pass 4: Remaining high-order harmonic centers
    if n_fft > n_priority:
        rem_orders = orders[n_priority:]
        rem_centers = centers[n_priority:]
        inside = (rem_centers >= 0) & (rem_centers <= n_half)
        pos = rem_centers[inside]
        ord_in = rem_orders[inside]
        free = owner[pos] == no_owner
        np.minimum.at(owner, pos[free], ord_in[free])

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
