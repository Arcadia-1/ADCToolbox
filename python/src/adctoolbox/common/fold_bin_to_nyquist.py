"""Calculate aliased bin index for FFT analysis.

Maps any bin index to the valid range [0, n_fft/2] for real signals.
Matches the naming convention of calculate_aliased_freq.
"""


def fold_bin_to_nyquist(bin_idx: int, n_fft: int) -> int:
    """
    Calculate the aliased bin index in the first Nyquist zone [0, n_fft/2].

    For real signals, FFT bins above n_fft/2 are mirrored to the first
    Nyquist zone. This function handles the wrapping and mirroring.

    Parameters
    ----------
    bin_idx : int
        Bin index (can be negative or > n_fft)
    n_fft : int
        Total number of FFT bins

    Returns
    -------
    int
        Aliased bin index in range [0, n_fft/2]

    Examples
    --------
    >>> fold_bin_to_nyquist(100, 8192)
    100
    >>> fold_bin_to_nyquist(5000, 8192)  # Above Nyquist, mirrors back
    3192
    >>> fold_bin_to_nyquist(-100, 8192)  # Negative wraps around
    92
    """
    # First wrap to [0, n_fft) range
    bin_idx = bin_idx % n_fft

    # For real signals, bins > n_fft/2 are mirrored
    if bin_idx > n_fft // 2:
        bin_idx = n_fft - bin_idx

    return bin_idx
