"""Window function creation and parameter calculation.

This module provides functions for creating window functions and calculating
their parameters:

- Window Gain (Coherent Gain): ``sum(window_vector) / N``
- Noise BW Factor (Equivalent Noise Bandwidth): ``N * sum(window_vector²) / sum(window_vector)²``

This is an internal helper module, not intended for direct use by end users.
"""

import numpy as np
from scipy.signal import windows


# Default side_bin values for each window's coherent main lobe.
# Non-coherent captures must pass a larger side_bin explicitly; inferring
# coherence from a distorted or quantized spectrum is too brittle.
_SIDE_BIN_DEFAULTS = {
    'rectangular': {
        'enbw': 1.00,
        'coherent': 0,
    },
    'hann': {
        'enbw': 1.50,
        'coherent': 1,
    },
    'hamming': {
        'enbw': 1.36,
        'coherent': 1,
    },
    'blackman': {
        'enbw': 1.73,
        'coherent': 2,
    },
    'blackmanharris': {
        'enbw': 2.00,
        'coherent': 3,
    },
    'flattop': {
        'enbw': 3.77,
        'coherent': 4,
    },
    'kaiser': {
        'enbw': 3.51,
        'coherent': 8,
    },
    'chebwin': {
        'enbw': 1.94,
        'coherent': 4,
    }
}


_SIDE_BIN_AUTO_FALLBACKS = {
    'rectangular': 1,
    'hann': 3,
    'hamming': 3,
    'blackman': 4,
    'blackmanharris': 5,
    'flattop': 6,
    'kaiser': 10,
    'chebwin': 6,
}


def _create_window(win_type: str, N: int) -> tuple[np.ndarray, float, float]:
    """Create window function and calculate its parameters.

    Parameters
    ----------
    win_type
        Window type: 'boxcar', 'rectangular', 'hann', 'hamming', 'blackman',
        'blackmanharris', 'flattop', 'kaiser', 'chebwin', etc.
    N
        Window length (number of samples)

    Returns
    -------
    window_vector
        Window function array, shape (N,)
    window_gain
        Amplitude scaling factor for single-tone signals
    noise_bw_factor
        Noise bandwidth factor for broadband noise
    """
    win_lower = win_type.lower()

    if win_lower in ('boxcar', 'rectangular'):
        window_vector = np.ones(N)
    elif win_lower == 'kaiser':
        window_vector = windows.kaiser(N, beta=38, sym=False)
    elif win_lower == 'chebwin':
        window_vector = windows.chebwin(N, at=100, sym=False)
    elif win_lower in ('hann', 'hamming'):
        win_func = getattr(windows, win_lower)
        window_vector = win_func(N, sym=False)
    elif win_lower in ('blackman', 'blackmanharris'):
        win_func = getattr(windows, win_lower)
        window_vector = win_func(N, sym=False)
    elif win_lower == 'flattop':
        window_vector = windows.flattop(N, sym=False)
    else:
        # Default to hann if window type not recognized
        window_vector = windows.hann(N, sym=False)

    window_gain = np.sum(window_vector) / N

    equiv_noise_bw_factor = N * np.sum(window_vector**2) / (np.sum(window_vector)**2)

    return window_vector, window_gain, equiv_noise_bw_factor


def _get_default_side_bin(win_type: str, is_coherent: bool | None = None) -> int:
    """Get the default coherent side_bin value for a window type.

    Parameters
    ----------
    win_type : str
        Window type name
    is_coherent : bool, optional
        Ignored. Kept only for internal backward compatibility with older
        callers. Non-coherent captures must pass ``side_bin`` explicitly.

    Returns
    -------
    int
        Default coherent side_bin value for the window type.
    """
    # Normalize window type (handle aliases)
    win_key = win_type.lower()
    if win_key == 'boxcar':
        win_key = 'rectangular'

    # Default to hann if window type not recognized
    if win_key not in _SIDE_BIN_DEFAULTS:
        win_key = 'hann'

    return _SIDE_BIN_DEFAULTS[win_key]['coherent']


def _get_auto_side_bin_fallback(win_type: str) -> int:
    """Small conservative fallback when side_bin auto cannot find a crossing."""
    win_key = win_type.lower()
    if win_key == 'boxcar':
        win_key = 'rectangular'

    if win_key not in _SIDE_BIN_AUTO_FALLBACKS:
        win_key = 'hann'

    return _SIDE_BIN_AUTO_FALLBACKS[win_key]


def _calculate_power_correction(window_gain: float, equiv_noise_bw_factor: float) -> float:
    """Calculate MATLAB plotspec-style power correction for dBFS scaling.

    The correction is equivalent to applying the window with RMS
    normalization and then multiplying the one-sided FFT power by 4. For a
    coherent Hann-windowed full-scale sine, this leaves the main tone split
    across the center bin and adjacent side bins; the main-lobe sum is 0 dBFS,
    while the center bin alone is below 0 dBFS.

    Since ``ENBW = mean(window²) / window_gain²``, the raw-window correction is
    ``4 / mean(window²) = 4 / (window_gain² * ENBW)``.

    Parameters
    ----------
    window_gain
        Window gain from _create_window
    equiv_noise_bw_factor
        Equivalent noise bandwidth factor from _create_window

    Returns
    -------
    float
        Power correction factor: ``4 / (window_gain² * ENBW)``
    """
    return 4 / (window_gain**2 * equiv_noise_bw_factor)
