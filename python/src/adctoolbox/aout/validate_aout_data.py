"""Validate analog output data format."""

import numpy as np


def validate_aout_data(aout_data):
    """
    Validate analog output data format.

    Parameters
    ----------
    aout_data : array_like
        Analog output signal data

    Raises
    ------
    ValueError
        If data is invalid with descriptive message
    """
    aout_data = np.asarray(aout_data)

    if not np.issubdtype(aout_data.dtype, np.number):
        raise ValueError(f'Data must be numeric, got {aout_data.dtype}')

    if np.iscomplexobj(aout_data):
        raise ValueError('Data must be real-valued, got complex numbers')

    if np.any(np.isnan(aout_data)):
        raise ValueError('Data contains NaN values')

    if np.any(np.isinf(aout_data)):
        raise ValueError('Data contains Inf values')

    if aout_data.size == 0:
        raise ValueError('Data is empty')

    if aout_data.ndim == 1:
        n_samples = len(aout_data)
    else:
        n_samples = aout_data.shape[1]

    if n_samples < 100:
        raise ValueError(f'Insufficient samples ({n_samples}), need at least 100')

    data_range = np.max(aout_data) - np.min(aout_data)
    if data_range == 0:
        raise ValueError('Data is constant (no variation)')

    if data_range < 1e-10:
        raise ValueError(f'Data range too small ({data_range:.2e}), likely invalid')
