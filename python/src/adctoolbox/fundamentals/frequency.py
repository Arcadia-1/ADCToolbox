"""
Frequency Utilities for ADC Testing.

This module provides frequency-related calculations for coherent sampling,
aliasing, FFT bin folding, and frequency estimation in ADC characterization.
"""

import math
import numpy as np
from adctoolbox.fundamentals.fit_sine_4param import fit_sine_4param

def find_coherent_frequency(fs, fin_target, n_fft, force_odd=True, search_radius=200):
    """
    Calculate the precise coherent input frequency and bin index.

    Supports Undersampling (Fin > Fs/2).

    Parameters
    ----------
    fs : float
        Sampling frequency (Hz)
    fin_target : float
        Target input frequency (Hz)
    n_fft : int
        FFT size (number of points)
    force_odd : bool, optional
        If True, only search for odd bin indices (default: True)
    search_radius : int, optional
        Search radius around the ideal bin (default: 200)

    Returns
    -------
    tuple
        (fin_actual, best_bin) - Coherent frequency and corresponding bin index

    Raises
    ------
    ValueError
        If no valid coherent frequency is found within search radius

    Examples
    --------
    >>> fin, bin_idx = find_coherent_frequency(800e6, 100e6, 8192)
    >>> print(f"Coherent frequency: {fin/1e6:.6f} MHz at bin {bin_idx}")
    """
    # 1. Calculate the ideal (fractional) total cycles
    target_bin_float = fin_target / fs * n_fft

    # 2. Define search center (nearest integer)
    center_int = int(round(target_bin_float))

    candidates = []

    # 3. Search neighborhood for the best candidate
    for i in range(-search_radius, search_radius + 1):
        bin_candidate = center_int + i

        # Validity checks: Only check if positive.
        # REMOVED: "or bin_candidate >= n_fft // 2" to allow undersampling/high freq.
        if bin_candidate <= 0:
            continue

        # Condition A: Force Odd (Standard practice)
        if force_odd and (bin_candidate % 2 == 0):
            continue

        # Condition B: Coprime Check (GCD == 1)
        # Even if M > N, if they are coprime, the aliased bin will be unique.
        if math.gcd(bin_candidate, n_fft) == 1:
            # Score by distance to target
            dist = abs(bin_candidate - target_bin_float)
            candidates.append((dist, bin_candidate))

    if not candidates:
        raise ValueError(
            f"No valid prime/odd bin found near {target_bin_float:.2f} cycles "
            f"(Fin={fin_target/1e6:.2f}MHz). Try increasing search_radius."
        )

    # 4. Pick the winner (smallest distance)
    candidates.sort(key=lambda x: x[0])
    best_bin = candidates[0][1]

    # 5. Calculate precise physical frequency
    fin_actual = best_bin * fs / n_fft

    return fin_actual, best_bin

def fold_frequency_to_nyquist(fin, fs):
    """
    Calculate the aliased (folded) frequency in the first Nyquist zone.

    The aliased frequency is the absolute difference between the input frequency
    and the nearest integer multiple of the sampling rate.

    Parameters
    ----------
    fin : float or np.ndarray
        Input frequency (Hz). Can be positive or negative.
    fs : float
        Sampling frequency (Hz)

    Returns
    -------
    float or np.ndarray
        Aliased frequency in range [0, Fs/2]

    Examples
    --------
    >>> fold_frequency_to_nyquist(100e6, 800e6)
    100000000.0
    >>> fold_frequency_to_nyquist(900e6, 800e6)
    100000000.0
    """
    fin = np.asarray(fin)

    # The mathematical one-liner for folding:
    # F_alias = | Fin - Fs * round(Fin / Fs) |
    # Note: np.round rounds to nearest even number for .5 cases, which is fine here.

    f_alias = np.abs(fin - fs * np.round(fin / fs))

    # Edge case handling: If logic results in exactly Fs/2, it's correct.
    # If fin was already scalar, convert back to scalar for cleaner return types (optional)
    if fin.ndim == 0:
        return float(f_alias)

    return f_alias

def fold_bin_to_nyquist(bin_idx: float, n_fft: int) -> float:
    """
    Calculate the aliased bin index in the first Nyquist zone [0, n_fft/2].

    For real signals, FFT bins above n_fft/2 are mirrored to the first
    Nyquist zone. This function handles the wrapping and mirroring.

    Parameters
    ----------
    bin_idx : float
        Bin index (can be fractional, negative, or > n_fft)
    n_fft : int
        Total number of FFT bins

    Returns
    -------
    float
        Aliased bin index in range [0, n_fft/2]

    Examples
    --------
    >>> fold_bin_to_nyquist(100, 8192)
    100.0
    >>> fold_bin_to_nyquist(100.5, 8192)  # Fractional bins supported
    100.5
    >>> fold_bin_to_nyquist(5000, 8192)  # Above Nyquist, mirrors back
    3192.0
    >>> fold_bin_to_nyquist(-100, 8192)  # Negative wraps around
    92.0
    """
    # First wrap to [0, n_fft) range
    bin_idx = bin_idx % n_fft

    # For real signals, bins > n_fft/2 are mirrored
    if bin_idx > n_fft // 2:
        bin_idx = n_fft - bin_idx

    return bin_idx

def estimate_frequency(data, frequency_estimate=None, fs=1.0):
    """
    Estimate the physical fundamental frequency (Hz) of a signal.

    This is a wrapper around the robust `fit_sine_4param` algorithm.
    It converts the normalized frequency (0 ~ 0.5) returned by fit_sine
    into physical frequency (Hz) based on the sampling rate.

    Parameters
    ----------
    data : np.ndarray
        Input signal data. 1D or 2D array.
    fs : float, optional
        Sampling frequency in Hz (default: 1.0)

    Returns
    -------
    float or np.ndarray
        Estimated frequency in Hz.
        (Scalar if input is 1D, Array if input is 2D)

    Examples
    --------
    >>> import numpy as np
    >>> # Generate a 100 MHz sine wave sampled at 800 MHz
    >>> t = np.arange(8192) / 800e6
    >>> signal = np.sin(2 * np.pi * 100e6 * t)
    >>> freq = estimate_frequency(signal, fs=800e6)
    >>> print(f"Estimated frequency: {freq/1e6:.2f} MHz")
    Estimated frequency: 100.00 MHz
    """
    result = fit_sine_4param(data, frequency_estimate=frequency_estimate)

    freq_norm = result['frequency']

    fin_hz = freq_norm * fs

    return fin_hz
