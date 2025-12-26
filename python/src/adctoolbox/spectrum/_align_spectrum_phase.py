"""
Align spectrum phase for coherent averaging.

This module implements the phase alignment algorithm for FFT coherent mode.
"""

import numpy as np

def _align_spectrum_phase(fft_data: np.ndarray, bin_idx: int, bin_r: float, n_fft: int) -> np.ndarray:
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

    Returns
    -------
    np.ndarray
        Phase-aligned FFT spectrum (complex)

    Notes
    -----
    Algorithm:
    - Harmonics are rotated by integer multiples of the fundamental phase
    - Nyquist zone detection: even zones use normal phase, odd zones use conjugate
    - Non-harmonics are rotated by fractional phase based on bin position
    - DC bin is zeroed after alignment
    """
    # Guard against DC bin
    if bin_idx <= 0:
        return fft_data

    # Calculate phase rotation to align fundamental to phase 0
    fundamental_phasor = fft_data[bin_idx] / (np.abs(fft_data[bin_idx]) + 1e-20)
    phasor = np.conj(fundamental_phasor)

    # Create copy for output
    fft_aligned = fft_data.copy()

    # Track which bins have been processed
    processed = np.zeros(n_fft, dtype=bool)

    # Step 1: Align harmonics with Nyquist folding
    harmonic_phasor = phasor

    for harmonic_order in range(1, n_fft + 1):
        # Calculate harmonic frequency
        harmonic_freq = bin_idx * harmonic_order

        # Determine Nyquist zone (even or odd)
        nyquist_zone = np.floor(harmonic_freq / n_fft * 2)
        is_even_zone = (nyquist_zone % 2) == 0

        if is_even_zone:
            # Even zone: normal aliasing
            bin_pos = int(harmonic_freq - np.floor(harmonic_freq / n_fft) * n_fft)

            if not processed[bin_pos]:
                fft_aligned[bin_pos] *= harmonic_phasor
                processed[bin_pos] = True
        else:
            # Odd zone: mirrored aliasing (use conjugate)
            bin_pos = int(n_fft - harmonic_freq + np.floor(harmonic_freq / n_fft) * n_fft)

            if 0 <= bin_pos < n_fft and not processed[bin_pos]:
                fft_aligned[bin_pos] *= np.conj(harmonic_phasor)
                processed[bin_pos] = True

        # Accumulate phase for next harmonic
        harmonic_phasor *= phasor

    # Step 2: Align non-harmonics with fractional phase
    for bin_pos in range(n_fft):
        if not processed[bin_pos]:
            # Apply fractional phase shift based on bin position relative to refined fundamental
            fractional_phase = bin_pos / bin_r if bin_r > 0 else 0
            fft_aligned[bin_pos] *= phasor ** fractional_phase

    # Zero out DC component
    fft_aligned[0] = 0

    return fft_aligned
