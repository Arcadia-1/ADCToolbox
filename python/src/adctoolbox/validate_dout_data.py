"""Validate digital output (bits) data format."""

import numpy as np
import warnings


def validate_dout_data(bits):
    """
    Validate digital output (bits) data format.

    Parameters
    ----------
    bits : array_like
        Digital bits (N samples x B bits, MSB to LSB)

    Raises
    ------
    ValueError
        If data is invalid with descriptive message
    """
    bits = np.asarray(bits)

    if not np.issubdtype(bits.dtype, np.number):
        raise ValueError(f'Data must be numeric, got {bits.dtype}')

    if np.iscomplexobj(bits):
        raise ValueError('Data must be real-valued, got complex numbers')

    if np.any(np.isnan(bits)):
        raise ValueError('Data contains NaN values')

    if np.any(np.isinf(bits)):
        raise ValueError('Data contains Inf values')

    if bits.size == 0:
        raise ValueError('Data is empty')

    if bits.ndim != 2:
        raise ValueError(f'Data must be 2D matrix (N samples x B bits), got shape {bits.shape}')

    unique_vals = np.unique(bits)
    if not np.all(np.isin(unique_vals, [0, 1])):
        raise ValueError(f'Data must contain only binary values (0 or 1), found values: {unique_vals}')

    n_samples, n_bits = bits.shape

    if n_samples < 100:
        raise ValueError(f'Insufficient samples ({n_samples}), need at least 100')

    if n_bits < 2:
        raise ValueError(f'Insufficient bits ({n_bits}), need at least 2')

    if n_bits > 32:
        warnings.warn(f'Unusual bit count ({n_bits}), verify this is correct')

    stuck_bits = np.sum(bits, axis=0)
    all_zero_bits = np.where(stuck_bits == 0)[0]
    all_one_bits = np.where(stuck_bits == n_samples)[0]

    if len(all_zero_bits) > 0:
        warnings.warn(f'Bit(s) stuck at 0: {all_zero_bits}')

    if len(all_one_bits) > 0:
        warnings.warn(f'Bit(s) stuck at 1: {all_one_bits}')
