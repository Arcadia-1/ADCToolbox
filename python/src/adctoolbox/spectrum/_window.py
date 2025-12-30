"""Window function creation and parameter calculation.

This module provides functions for creating window functions and calculating
their parameters:

- Window Gain (Coherent Gain): ``sum(window_vector) / N``
- Noise BW Factor (Equivalent Noise Bandwidth): ``N * sum(window_vector²) / sum(window_vector)²``

This is an internal helper module, not intended for direct use by end users.
"""

import numpy as np
from scipy.signal import windows


# Default side_bin values for window functions based on signal coherence
# Coherent: Empirically tuned to maintain ENOB > 13b
# Non-coherent: Empirically tuned to maintain ENOB > 12b
_SIDE_BIN_DEFAULTS = {
    'rectangular': {
        'enbw': 1.00,
        'coherent': 0,
        'non_coherent': 10
    },
    'hann': {
        'enbw': 1.50,
        'coherent': 1,
        'non_coherent': 10
    },
    'hamming': {
        'enbw': 1.36,
        'coherent': 1,
        'non_coherent': 10
    },
    'blackman': {
        'enbw': 1.73,
        'coherent': 2,
        'non_coherent': 10
    },
    'blackmanharris': {
        'enbw': 2.00,
        'coherent': 3,
        'non_coherent': 5
    },
    'flattop': {
        'enbw': 3.77,
        'coherent': 4,
        'non_coherent': 5
    },
    'kaiser': {
        'enbw': 3.51,
        'coherent': 8,
        'non_coherent': 15
    },
    'chebwin': {
        'enbw': 1.94,
        'coherent': 4,
        'non_coherent': 5
    }
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


def _get_default_side_bin(win_type: str, is_coherent: bool) -> int:
    """Get default side_bin value for a window type based on coherence.

    Parameters
    ----------
    win_type : str
        Window type name
    is_coherent : bool
        True if signal is coherent (coherence_error < 0.01), False otherwise

    Returns
    -------
    int
        Default side_bin value for the window type and coherence state
    """
    # Normalize window type (handle aliases)
    win_key = win_type.lower()
    if win_key == 'boxcar':
        win_key = 'rectangular'

    # Default to hann if window type not recognized
    if win_key not in _SIDE_BIN_DEFAULTS:
        win_key = 'hann'

    # Return appropriate side_bin value
    key = 'coherent' if is_coherent else 'non_coherent'
    return _SIDE_BIN_DEFAULTS[win_key][key]


def _calculate_power_correction(window_gain: float) -> float:
    """Calculate power correction factor for dBFS scaling.

    The factor 4 comes from:
    - 2x for single-sided spectrum (positive frequencies only)
    - 2x for peak-to-RMS conversion for power scaling

    Division by window_gain² accounts for window power normalization.
    This ensures a full-scale sine (peak=1) normalizes to 0 dBFS.

    Parameters
    ----------
    window_gain
        Window gain from _create_window

    Returns
    -------
    float
        Power correction factor: ``4 / window_gain²``
    """
    return 4 / window_gain**2
