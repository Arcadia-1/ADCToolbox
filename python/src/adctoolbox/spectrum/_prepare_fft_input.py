"""Prepare FFT input data - shared helper for spectrum analysis."""

import numpy as np
import warnings
from scipy.signal import windows
from typing import Optional, Union, Tuple, List


def _prepare_fft_input(
    data: np.ndarray,
    max_scale_range: Optional[Union[float, Tuple[float, float], List[float]]] = None,
    win_type: str = 'boxcar'
) -> np.ndarray:
    """Prepare input data for FFT analysis.

    Parameters
    ----------
    data : np.ndarray
        Input ADC data, shape (N,) or (M, N). Standard format: (M runs, N samples).
        Auto-transposes if N >> M (with warning).
    max_scale_range : float, tuple, list, or None, optional
        Full scale range for normalization. Can be specified as:
        - None: auto-detect as (max - min) from data
        - float: direct full-scale range magnitude
        - tuple/list of 2 floats: (min, max) ADC range, range magnitude = max - min
        Default: None
    win_type : str, optional
        Window type: 'boxcar', 'hann', 'hamming', etc. Default: 'boxcar'.

    Returns
    -------
    np.ndarray
        Processed data ready for FFT, shape (M, N).
    """
    # Ensure 2D array
    data = np.atleast_2d(data)
    if data.ndim > 2:
        raise ValueError(f"Input must be 1D or 2D, got {data.ndim}D")

    n_rows, n_cols = data.shape

    # Auto-transpose to standard (M runs, N samples) format
    if n_cols == 1 and n_rows > 1:
        data = data.T  # (N, 1) -> (1, N)
    elif n_rows > 1 and n_cols > 1 and n_rows > n_cols * 2:
        warnings.warn(f"[Auto-transpose] Input shape [{n_rows}, {n_cols}] -> [{n_cols}, {n_rows}]. Standard format is (M runs, N samples).", UserWarning, stacklevel=3)
        data = data.T

    # M = number of runs, N = number of fft points = number of samples
    M, N = data.shape

    # Normalization scale
    # When auto-detecting (None), use full-scale range (max - min)
    # dBFS reference: full-scale sine (peak=1) has power=0.5 = 0 dBFS
    if max_scale_range is None:
        max_scale_range = np.max(data) - np.min(data)
    elif isinstance(max_scale_range, (list, tuple)):
        # Convert (min, max) range to magnitude
        if len(max_scale_range) != 2:
            raise ValueError(f"Range tuple/list must have 2 elements, got {len(max_scale_range)}")
        max_scale_range = max_scale_range[1] - max_scale_range[0]

    # Create window function
    if win_type.lower() in ('boxcar', 'rectangular'):
        # Rectangular window
        win = np.ones(N)
    elif win_type.lower() == 'kaiser':
        # Kaiser window requires beta parameter (38 for very high 
        # side lobe suppression)
        win = windows.kaiser(N, beta=38, sym=False)
    elif win_type.lower() == 'chebwin':
        # Chebyshev window requires attenuation parameter (100 dB typical)
        win = windows.chebwin(N, at=100, sym=False)
    else:
        # Other window types from scipy.signal.windows, if not found default to Hann
        win_func = getattr(windows, win_type.lower(), windows.hann)
        win = win_func(N, sym=False)  # Periodic window for FFT

    # Power-normalize window
    win_normalized = win / np.sqrt(np.mean(win**2))

    # Vectorized processing: DC removal -> normalization -> windowing
    data_dc_removed = data - np.mean(data, axis=1, keepdims=True)
    data_normalized = data_dc_removed / max_scale_range if max_scale_range != 0 else data_dc_removed
    processed_data = data_normalized * win_normalized

    return processed_data
