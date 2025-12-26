"""
Least Squares Sine Wave Fitting (IEEE Std 1057/1241).

Supports 3-parameter (frequency-fixed) and 4-parameter (frequency-optimized) fitting modes.
"""

import numpy as np

def fit_sine_4param(data, frequency_estimate=None, max_iterations=1, tolerance=1e-9):
    """
    Fit sine wave: y = A*cos(wt) + B*sin(wt) + C.

    Args:
        data: Input signal (1D or 2D array).
        frequency_estimate: Initial normalized frequency (0 to 0.5). If None, estimated via FFT.
        max_iterations: Iterations for frequency refinement.
        tolerance: Convergence threshold for frequency updates.

    Returns:
        Dictionary with fitted parameters:
            - fitted_signal: Reconstructed sine wave
            - residuals: Data - fitted_signal
            - frequency: Normalized frequency (0 to 0.5)
            - amplitude: sqrt(A² + B²)
            - phase: atan2(-B, A) in radians
            - dc_offset: DC component
            - rmse: Root mean square error

        For 2D input, all values (except fitted_signal, residuals) are 1D arrays.
    """
    data = np.asarray(data)

    if data.ndim == 1:
        return _fit_core(data, frequency_estimate, max_iterations, tolerance)

    if data.ndim == 2:
        results = [_fit_core(ch, frequency_estimate, max_iterations, tolerance) for ch in data.T]
        return _merge_results(results)

    raise ValueError(f"Input must be 1D or 2D, got {data.ndim}D.")

def _fit_core(y, freq_init, max_iter, tol):
    """Fit sine wave to 1D signal using least squares."""
    n = len(y)
    t = np.arange(n)
    freq = freq_init if freq_init is not None else _estimate_frequency_fft(y)
    a = b = c = 0.0

    for i in range(max_iter + 1):
        omega = 2 * np.pi * freq
        cos_vec = np.cos(omega * t)
        sin_vec = np.sin(omega * t)

        # Build design matrix: 3-param on iteration 0, 4-param on later iterations
        if i == 0:
            design_matrix = np.column_stack((cos_vec, sin_vec, np.ones(n)))
        else:
            freq_corr = t * (-a * sin_vec + b * cos_vec)
            design_matrix = np.column_stack((cos_vec, sin_vec, np.ones(n), freq_corr))

        coeffs = np.linalg.lstsq(design_matrix, y, rcond=None)[0]
        a, b, c = coeffs[:3]

        if len(coeffs) > 3:
            delta_freq = coeffs[3] / (2 * np.pi)
            freq += delta_freq
            if abs(delta_freq) < tol:
                break

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

def _merge_results(results_list):
    """Merge list of result dicts into single dict with stacked arrays."""
    merged = {}
    for key in results_list[0].keys():
        values = [r[key] for r in results_list]
        if np.ndim(values[0]) == 0:  # Scalar
            merged[key] = np.array(values)
        else:  # Array (fitted_signal, residuals)
            merged[key] = np.column_stack(values)
    return merged

def _estimate_frequency_fft(y):
    """Estimate frequency using FFT peak with parabolic interpolation."""
    n = len(y)
    spec = np.abs(np.fft.fft(y))
    spec[0] = 0
    spec = spec[:n // 2]
    k = np.argmax(spec)

    # Parabolic interpolation for sub-bin precision
    if 0 < k < len(spec) - 1:
        r = 1 if spec[k + 1] > spec[k - 1] else -1
        delta = r * spec[k + r] / (spec[k] + spec[k + r])
        k += delta

    return k / n
