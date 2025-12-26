"""Minimal ADC calibration using sinewave input at known frequency."""

import numpy as np
from scipy.linalg import lstsq

def calibrate_weight_sine_lite(
    bits: np.ndarray,
    freq: float
) -> np.ndarray:
    """
    Minimal calibration at known frequency. Returns normalized weights.
    Expects well-conditioned binary data (N samples x M bits).
    """
    n_samples, bit_width = bits.shape

    # Build fundamental basis
    t = np.arange(n_samples)
    phase = 2.0 * np.pi * freq * t
    cos_basis = np.cos(phase)
    sin_basis = np.sin(phase)

    # Cosine=1 assumption
    offset_col = np.ones((n_samples, 1))
    A = np.column_stack([bits, offset_col, sin_basis])
    b = -cos_basis
    coeffs, _, _, _ = lstsq(A, b)

    # Extract weights and normalization
    weights_raw = coeffs[:bit_width]
    sin_coeff = coeffs[-1]
    norm_factor = np.sqrt(1.0 + sin_coeff**2)
    weights = weights_raw / norm_factor

    # Polarity correction (ensure positive sum)
    if np.sum(weights) < 0:
        weights = -weights

    return weights
