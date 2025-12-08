

import numpy as np


def _align_spectrum_phase(fft_data: np.ndarray, bin_idx: int, bin_r: float, n_fft: int) -> np.ndarray:
    """
    Align the phase of an FFT spectrum to phase 0 at the fundamental frequency.

    This function performs phase rotation to align harmonics and non-harmonics
    separately, matching MATLAB plotphase.m FFT mode logic.

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
    - Harmonics are rotated by integer multiples of the fundamental phase
    - Non-harmonics are rotated by fractional phase based on their bin position
    - DC bin is zeroed after alignment
    """
    # Calculate phase rotation to align fundamental to phase 0
    fundamental_phase = np.angle(fft_data[bin_idx])
    phase_rotation = np.exp(-1j * fundamental_phase)

    # Apply phase alignment to entire spectrum
    fft_aligned = fft_data.copy()

    # Phase shift harmonics and non-harmonics separately
    # This matches MATLAB plotphase.m FFT mode logic
    marker = np.zeros(n_fft, dtype=bool)

    for h in range(1, n_fft):
        # Calculate harmonic number
        harmonic_num = h

        # Determine if this is a harmonic of the fundamental
        harmonic_bin = (bin_idx * harmonic_num) % n_fft

        # Handle aliasing for real signals
        if harmonic_bin > n_fft // 2:
            harmonic_bin = n_fft - harmonic_bin

        # Calculate appropriate phase shift
        phase_shift = phase_rotation ** (harmonic_num if not marker[harmonic_bin] else (h / bin_r))

        if not marker[harmonic_bin]:
            # This is a harmonic bin
            fft_aligned[harmonic_bin] *= phase_shift
            marker[harmonic_bin] = True
        else:
            # This is a non-harmonic bin
            if not marker[h]:
                fft_aligned[h] *= phase_shift ** ((h) / bin_r)
                marker[h] = True

    # Remove DC
    fft_aligned[0] = 0

    return fft_aligned
