"""
Align spectrum phase for coherent averaging.

This module implements the phase alignment algorithm for FFT coherent mode,
matching MATLAB plotphase.m and plotspec.m (coherent mode) exactly.
"""

import numpy as np


def _align_spectrum_phase(fft_data: np.ndarray, bin_idx: int, bin_r: float, n_fft: int) -> np.ndarray:
    """
    Align the phase of an FFT spectrum to phase 0 at the fundamental frequency.

    This function performs phase rotation to align harmonics and non-harmonics
    separately, matching MATLAB plotphase.m FFT mode logic exactly.

    The algorithm uses two separate loops:
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
    Algorithm matches MATLAB plotspec.m lines 290-322 and plotphase.m lines 154-180:
    - Harmonics are rotated by integer multiples of the fundamental phase
    - Nyquist zone detection: even zones use normal phase, odd zones use conjugate
    - Non-harmonics are rotated by fractional phase based on (bin-1)/(bin_r-1)
    - DC bin is zeroed after alignment
    """
    # Guard against DC bin
    if bin_idx <= 0:
        return fft_data

    # Calculate phase rotation to align fundamental to phase 0
    # MATLAB: phi = tspec(bin)/abs(tspec(bin))
    fundamental_phasor = fft_data[bin_idx] / (np.abs(fft_data[bin_idx]) + 1e-20)

    # MATLAB: phasor = conj(phi)
    phasor = np.conj(fundamental_phasor)

    # Create copy for output
    fft_aligned = fft_data.copy()

    # Marker array to track which bins have been processed
    marker = np.zeros(n_fft, dtype=bool)

    # ========== LOOP 1: Harmonic phase shift with Nyquist folding ==========
    # MATLAB plotspec.m lines 296-315, plotphase.m lines 158-174
    phasor_accumulator = phasor

    for h in range(1, n_fft + 1):
        # Calculate harmonic frequency (MATLAB: J = (bin-1)*iter2)
        # Note: MATLAB uses 1-based indexing, Python uses 0-based
        J = (bin_idx) * h  # bin_idx is already 0-based

        # Determine if harmonic is in even or odd Nyquist zone
        # MATLAB: if(mod(floor(J/N_fft*2),2) == 0)
        nyquist_zone = np.floor(J / n_fft * 2)
        is_even_zone = (nyquist_zone % 2) == 0

        if is_even_zone:
            # Even zone: normal aliasing
            # MATLAB: b = J-floor(J/N_fft)*N_fft+1 (1-based)
            # Python: b = J - floor(J/N_fft)*N_fft (0-based)
            b = int(J - np.floor(J / n_fft) * n_fft)

            if not marker[b]:
                # MATLAB: tspec(b) = tspec(b).*phasor
                fft_aligned[b] = fft_aligned[b] * phasor_accumulator
                marker[b] = True
        else:
            # Odd zone: mirrored aliasing (conjugate)
            # MATLAB: b = N_fft-J+floor(J/N_fft)*N_fft+1 (1-based)
            # Python: b = N_fft-J+floor(J/N_fft)*N_fft (0-based, then -1 for index)
            b = int(n_fft - J + np.floor(J / n_fft) * n_fft)

            if 0 <= b < n_fft and not marker[b]:
                # MATLAB: tspec(b) = tspec(b).*conj(phasor)
                fft_aligned[b] = fft_aligned[b] * np.conj(phasor_accumulator)
                marker[b] = True

        # MATLAB: phasor = phasor * conj(phi)
        phasor_accumulator = phasor_accumulator * phasor

    # ========== LOOP 2: Non-harmonic phase shift ==========
    # MATLAB plotspec.m lines 318-322, plotphase.m lines 176-180
    for k in range(n_fft):
        if not marker[k]:
            # Apply fractional phase shift based on bin position
            # MATLAB plotspec.m:320: tspec(iter2) = tspec(iter2).*(conj(phi).^((iter2-1)/(bin-1)))
            # MATLAB plotphase.m:178: tspec(iter2) = tspec(iter2).*(conj(phi).^((iter2-1)/(bin_r-1)))
            # Note: bin_r is the refined bin location
            fractional_phase = (k) / (bin_r) if bin_r > 0 else 0
            fft_aligned[k] = fft_aligned[k] * (phasor ** fractional_phase)

    # Remove DC component
    fft_aligned[0] = 0

    return fft_aligned
