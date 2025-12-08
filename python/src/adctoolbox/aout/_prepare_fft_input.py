"""Prepare FFT input data - shared helper for spectrum analysis.

This module handles common preprocessing steps for FFT-based ADC analysis:
- DC removal
- Normalization to full scale
- Window function application
- Multi-run data handling

This is an internal helper module, not intended for direct use by end users.
"""

import numpy as np
from scipy.signal import windows
from typing import Tuple, Optional, Union


def _prepare_fft_input(
    data: np.ndarray,
    max_code: Optional[float] = None,
    win_type: str = 'boxcar',
    n_fft: Optional[int] = None
) -> Tuple[np.ndarray, float, int]:
    """Prepare input data for FFT analysis.

    Performs DC removal, normalization, and windowing operations that are
    common to both spectrum metrics calculation and coherent spectrum analysis.

    Parameters
    ----------
    data : np.ndarray
        Input ADC data, shape (N,) for single run or (M, N) for M runs
    max_code : float, optional
        Maximum code level for normalization. If None, uses (max(data) - min(data))
    win_type : str, optional
        Window function type: 'boxcar', 'hann', 'hamming', etc.
        Default is 'boxcar' (rectangular window)
    n_fft : int, optional
        FFT length. If None, uses the length of the data

    Returns
    -------
    processed_data : np.ndarray
        Processed data ready for FFT, same shape as input
    max_code_used : float
        The max_code value that was used for normalization
    n_samples : int
        Number of samples per run (N)

    Notes
    -----
    - Handles both single run and multi-run data
    - Removes DC offset from each run
    - Normalizes to full scale range
    - Applies window function with power normalization
    """
    # Convert to numpy array if needed
    data = np.asarray(data)

    # Handle different input shapes
    if data.ndim == 0:
        data = data.reshape(1, 1)
    elif data.ndim == 1:
        data = data.reshape(1, -1)
    elif data.ndim == 2:
        # Ensure shape is (M, N) not (N, M)
        M, N = data.shape
        if N == 1 and M > 1:
            data = data.T
    else:
        raise ValueError(f"Input data must be 1D or 2D, got {data.ndim}D")

    # Get dimensions
    M, N_samples = data.shape

    # Set n_fft if not provided
    if n_fft is None:
        n_fft = N_samples

    # Determine max_code for normalization
    if max_code is None:
        # Use peak-to-peak value (match MATLAB behavior)
        max_code = np.max(data) - np.min(data)

    # Create window function
    if win_type.lower() == 'boxcar' or win_type.lower() == 'rectangular':
        win = np.ones(N_samples)
    else:
        # Create symmetrical window for signal processing
        win_func = getattr(windows, win_type.lower(), windows.hann)
        if win_func == windows.hann:
            win = win_func(N_samples, sym=False)
        else:
            win = win_func(N_samples, sym=True)

    # Normalize window to preserve power
    win_power = np.sqrt(np.mean(win**2))

    # Process each run
    processed_data = np.zeros_like(data)

    for i in range(M):
        # Get current run
        run_data = data[i, :n_fft].copy()  # Truncate if needed

        # Remove DC offset
        run_data = run_data - np.mean(run_data)

        # Normalize to full scale
        if max_code != 0:
            run_data = run_data / max_code

        # Apply window function (power normalized)
        run_data = run_data * win / win_power

        processed_data[i, :len(run_data)] = run_data

    return processed_data, max_code, N_samples


def _get_window_correction(win_type: str) -> float:
    """Get amplitude correction factor for window function.

    Different windows have different amplitude scaling. This function returns
    the correction factor to compensate for the window's amplitude attenuation.

    Parameters
    ----------
    win_type : str
        Window function type

    Returns
    -------
    correction : float
        Amplitude correction factor
    """
    win_type = win_type.lower()

    # Correction factors for common windows
    corrections = {
        'boxcar': 1.0,
        'rectangular': 1.0,
        'hann': 2.0,  # Hann window has -6 dB amplitude attenuation
        'hanning': 2.0,
        'hamming': 1.85,  # Hamming has ~-5.35 dB attenuation
        'blackman': 2.38,  # Blackman has ~-7.53 dB attenuation
        'flattop': 4.64,  # Flattop has ~-13.33 dB attenuation
    }

    return corrections.get(win_type, 1.0)