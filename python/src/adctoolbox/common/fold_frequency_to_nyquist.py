"""
Aliased Frequency Calculator.

Calculates the folded frequency locations in the first Nyquist zone [0, Fs/2].
Vectorized to support both scalar and array inputs.
"""

import numpy as np


def calculate_aliased_freq(fin, fs):
    """
    Calculate the aliased (folded) frequency in the first Nyquist zone.

    The aliased frequency is the absolute difference between the input frequency
    and the nearest integer multiple of the sampling rate.

    Args:
        fin (float or np.ndarray): Input frequency (Hz). Can be positive or negative.
        fs (float): Sampling frequency (Hz).

    Returns:
        float or np.ndarray: Aliased frequency in range [0, Fs/2].
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
