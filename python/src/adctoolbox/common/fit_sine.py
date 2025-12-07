"""
Sine Wave Fitting Module.

Implements IEEE Std 1057/1241 Least Squares Sine-wave Fitting.
Supports both 3-parameter (fixed frequency) and 4-parameter (frequency optimization) modes.
"""

import numpy as np


def fit_sine(data, frequency_estimate=None, max_iterations=1, tolerance=1e-9):
    """
    Fit a sine wave to input data using Least Squares.

    Args:
        data (np.ndarray): Input signal. 
                           - 1D array (N samples)
                           - 2D array (N samples, M channels)
        frequency_estimate (float, optional): Initial normalized frequency (0.0-0.5).
        max_iterations (int): Max iterations for frequency refinement.
        tolerance (float): Convergence tolerance.

    Returns:
        dict: Fitting results.
            If input is 1D: Values are floats/scalars.
            If input is 2D: Values are 1D arrays (length M).
            
            Keys:
            - 'fitted_signal': (N,) or (N, M) array
            - 'residuals':     (N,) or (N, M) array
            - 'frequency':     float or (M,) array
            - 'amplitude':     float or (M,) array
            - 'phase':         float or (M,) array
            - 'dc_offset':     float or (M,) array
            - 'rmse':          float or (M,) array
    """
    data = np.asarray(data)

    # === 1. Handle Single Channel (1D) ===
    if data.ndim == 1:
        return _fit_core(data, frequency_estimate, max_iterations, tolerance)

    # === 2. Handle Multi-Channel (2D) - The Elegant Way ===
    elif data.ndim == 2:
        n_samples, n_channels = data.shape
        
        # Pre-allocate dictionaries for speed isn't necessary for small M, 
        # but collecting results cleanly is.
        
        # Iterate over columns using data.T (Transposed) which is pythonic
        # and stack the results automatically.
        results_list = [
            _fit_core(channel_data, frequency_estimate, max_iterations, tolerance) 
            for channel_data in data.T
        ]
        
        # Merge list of dicts -> dict of arrays (Structure of Arrays)
        # keys are same for all, so pick from first result
        keys = results_list[0].keys()
        
        merged_result = {}
        for k in keys:
            # Stack results: 
            # Scalars -> 1D Array
            # 1D Arrays (signals) -> 2D Matrix (stack along axis 1)
            values = [res[k] for res in results_list]
            
            if np.isscalar(values[0]) or np.ndim(values[0]) == 0:
                merged_result[k] = np.array(values)
            else:
                # For signals (N,), stack them to become (N, M)
                merged_result[k] = np.column_stack(values)
                
        return merged_result

    else:
        raise ValueError(f"Input data must be 1D or 2D, got {data.ndim}D.")


def _fit_core(y, freq_init, max_iter, tol):
    """Core fitting logic for a single 1D signal."""
    n = len(y)
    t = np.arange(n)

    # 1. Frequency Estimation
    if freq_init is None:
        freq = _estimate_frequency_fft(y)
    else:
        freq = freq_init

    # Initialize params
    a, b, c = 0.0, 0.0, 0.0

    # 2. Iterative Fitting
    for i in range(max_iter + 1):
        omega = 2 * np.pi * freq
        cos_vec = np.cos(omega * t)
        sin_vec = np.sin(omega * t)

        if i == 0:
            # 3-param fit: [cos, sin, 1]
            design_matrix = np.column_stack((cos_vec, sin_vec, np.ones(n)))
        else:
            # 4-param fit: [cos, sin, 1, correction]
            # D = t * (-A*sin + B*cos)
            freq_corr = t * (-a * sin_vec + b * cos_vec)
            design_matrix = np.column_stack((cos_vec, sin_vec, np.ones(n), freq_corr))

        coeffs, _, _, _ = np.linalg.lstsq(design_matrix, y, rcond=None)
        
        a, b, c = coeffs[0], coeffs[1], coeffs[2]

        if len(coeffs) > 3:
            delta_freq = coeffs[3] / (2 * np.pi)
            freq += delta_freq
            if abs(delta_freq) < tol:
                break

    # 3. Final Calculation
    omega = 2 * np.pi * freq
    fitted_sig = a * np.cos(omega * t) + b * np.sin(omega * t) + c
    residuals = y - fitted_sig
    
    return {
        'fitted_signal': fitted_sig,
        'residuals': residuals,
        'frequency': freq,
        'amplitude': np.sqrt(a**2 + b**2),
        'phase': np.arctan2(-b, a),
        'dc_offset': c,
        'rmse': np.sqrt(np.mean(residuals**2))
    }


def _estimate_frequency_fft(y):
    """Rough frequency estimation using FFT."""
    n = len(y)
    w = np.hanning(n)
    spec = np.abs(np.fft.rfft(y * w))
    spec[0] = 0
    k = np.argmax(spec)
    
    # Parabolic/Weighted interpolation
    if 0 < k < len(spec) - 1:
        denom = spec[k-1] + spec[k] + spec[k+1]
        delta = (spec[k+1] - spec[k-1]) / denom if denom else 0
        k += delta
        
    return k / n