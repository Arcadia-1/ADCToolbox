"""
Compute harmonic decomposition of signal.

Core computation function for decomposing ADC output into fundamental signal,
harmonic distortion, and other noise.
"""

import numpy as np


def compute_harmonic_decomposition(data, normalized_freq=None, order=10):
    """
    Compute harmonic decomposition of signal.

    Parameters
    ----------
    data : array_like
        ADC output data, 1D numpy array
    normalized_freq : float, optional
        Normalized frequency (f_in / f_sample), auto-detect if None
    order : int, default=10
        Harmonic order for fitting (fits fundamental + harmonics 2 through order)

    Returns
    -------
    dict : Dictionary containing:
        - 'fundamental_signal': Fundamental sinewave (DC + I/Q components)
        - 'total_error': Total error (data - fundamental_signal)
        - 'harmonic_error': Harmonic distortion (2nd through nth)
        - 'other_error': Residual error (data - all harmonics)
        - 'normalized_freq': Detected/used normalized frequency
        - 'data': Original input data
        - 'n_samples': Number of samples
    """
    # Prepare data
    data = np.asarray(data).flatten()
    n_samples = len(data)
    t = np.arange(n_samples)

    # Auto-detect frequency if not provided
    if normalized_freq is None or np.isnan(normalized_freq):
        try:
            from findFin import findFin
            normalized_freq = findFin(data)
        except ImportError:
            # FFT-based frequency detection
            spec = np.abs(np.fft.fft(data))
            spec[0] = 0
            normalized_freq = np.argmax(spec[:n_samples//2]) / n_samples
            print(f"Warning: findFin not found, using FFT detection: freq={normalized_freq:.6f}")

    # Compute fundamental (I/Q) components
    phase = t * normalized_freq * 2 * np.pi
    cos_basis = np.cos(phase)
    sin_basis = np.sin(phase)

    dc_offset = np.mean(data)
    weight_i = np.mean(cos_basis * data) * 2
    weight_q = np.mean(sin_basis * data) * 2
    fundamental_signal = dc_offset + cos_basis * weight_i + sin_basis * weight_q

    # Build harmonic basis matrix (vectorized)
    cos_matrix = np.array([np.cos(phase * (k + 1)) for k in range(order)]).T
    sin_matrix = np.array([np.sin(phase * (k + 1)) for k in range(order)]).T
    basis_matrix = np.column_stack([cos_matrix, sin_matrix])

    # Least squares fit for harmonics
    weights, *_ = np.linalg.lstsq(basis_matrix, data, rcond=None)
    signal_all = dc_offset + basis_matrix @ weights

    # Decompose errors
    total_error = data - fundamental_signal
    harmonic_error = signal_all - fundamental_signal
    other_error = data - signal_all

    return {
        'fundamental_signal': fundamental_signal,
        'total_error': total_error,
        'harmonic_error': harmonic_error,
        'other_error': other_error,
        'normalized_freq': normalized_freq,
        'data': data,
        'n_samples': n_samples,
        'order': order
    }
