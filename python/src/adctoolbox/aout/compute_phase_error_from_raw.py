"""
Compute phase error using raw data (high precision).

Core computation kernel for computing phase error metrics using raw (unbinned)
error data. This approach provides higher precision than binned methods.
"""

import numpy as np
from typing import Dict
from .decompose_harmonics import fit_sinewave_components


def compute_phase_error_from_raw(
    signal: np.ndarray,
    normalized_freq: float
) -> Dict[str, np.ndarray]:
    """
    Compute phase error using raw data (high precision).

    This function fits a fundamental sinewave to the input signal, computes
    the residual error, and performs AM/PM separation using least-squares
    fitting on the raw (unbinned) error data. This provides more precise
    estimates than the binned approach.

    Parameters
    ----------
    signal : np.ndarray
        Input signal, 1D numpy array.
    normalized_freq : float
        Normalized frequency (f/fs), range 0-0.5.

    Returns
    -------
    dict : Dictionary containing:
        - 'am_param' : float
            Amplitude modulation parameter (amplitude noise).
        - 'pm_param' : float
            Phase modulation parameter (phase noise in radians).
        - 'baseline' : float
            Baseline noise floor (constant error variance).
        - 'fitted_signal' : np.ndarray
            Reconstructed fundamental signal from fitting.
        - 'error' : np.ndarray
            Residual error (signal - fitted_signal).
        - 'phase' : np.ndarray
            Phase values for each sample in radians.
        - 'error_rms' : float
            Total RMS error across all samples.
        - 'fundamental_amplitude' : float
            Amplitude of fitted fundamental component.
        - 'dc_offset' : float
            DC offset of fitted signal.
        - 'normalized_freq' : float
            Input normalized frequency (echoed for convenience).

    Notes
    -----
    The raw approach fits AM/PM models directly to the squared error values
    at each sample:

        error²(φ) = am_param² * cos²(φ) + pm_param² * sin²(φ) + baseline

    This uses all available data points without binning, providing higher
    precision estimates but potentially more sensitive to outliers.

    The AM parameter represents amplitude noise (constant across phase),
    while the PM parameter represents phase jitter (varies with sin(φ)).

    Examples
    --------
    >>> sig = np.sin(2*np.pi*0.1*np.arange(1000)) + 0.01*np.random.randn(1000)
    >>> result = compute_phase_error_from_raw(sig, normalized_freq=0.1)
    >>> print(f"AM param: {result['am_param']:.3e}")
    >>> print(f"PM param: {result['pm_param']:.3e}")
    >>> print(f"Error RMS: {result['error_rms']:.3e}")
    """
    # Validate inputs
    signal = np.asarray(signal).flatten()
    n_samples = len(signal)

    if not (0 < normalized_freq < 0.5):
        raise ValueError(f"normalized_freq must be in range (0, 0.5), got {normalized_freq}")

    # Step 1: Fit fundamental sinewave (DC + cos + sin)
    W, fitted_signal, basis_matrix, phase = fit_sinewave_components(
        signal, freq=normalized_freq, order=1, include_dc=True
    )

    # Extract DC and fundamental amplitude
    dc_offset = W[0]
    cos_coeff = W[1]
    sin_coeff = W[2]
    fundamental_amplitude = np.sqrt(cos_coeff**2 + sin_coeff**2)

    # Step 2: Compute residual error
    error = signal - fitted_signal
    error_rms = np.sqrt(np.mean(error**2))

    # Step 3: Perform AM/PM separation using least-squares on raw error
    # Model: error²(φ) = am² * cos²(φ) + pm² * A² * sin²(φ) + baseline

    # Build sensitivity curves for each sample
    phase_wrapped = np.mod(phase, 2*np.pi)
    am_sensitivity = np.cos(phase_wrapped)**2  # Amplitude modulation sensitivity
    pm_sensitivity = np.sin(phase_wrapped)**2  # Phase modulation sensitivity

    # Square the error for fitting
    error_squared = error**2

    # Design matrix: [AM_sensitivity, PM_sensitivity, ones]
    A_matrix = np.column_stack([
        am_sensitivity,
        pm_sensitivity,
        np.ones(n_samples)
    ])

    # Solve least-squares: A @ [am², pm²*A², baseline] ≈ error²
    try:
        params, residuals, rank, s = np.linalg.lstsq(A_matrix, error_squared, rcond=None)
    except np.linalg.LinAlgError:
        # Fallback to simple estimates if fitting fails
        params = np.array([0.0, 0.0, np.mean(error_squared)])

    # Extract parameters with validation
    am_param_sq = params[0] if params[0] >= 0 else 0.0
    pm_param_sq = params[1] if params[1] >= 0 else 0.0
    baseline = params[2] if params[2] >= 0 else 0.0

    am_param = np.sqrt(am_param_sq)

    # PM parameter: sqrt(pm²*A²) / A = pm (in radians)
    if fundamental_amplitude > 1e-10:
        pm_param = np.sqrt(pm_param_sq) / fundamental_amplitude
    else:
        pm_param = 0.0

    # Return comprehensive results
    return {
        'am_param': float(am_param),
        'pm_param': float(pm_param),
        'baseline': float(baseline),
        'fitted_signal': fitted_signal,
        'error': error,
        'phase': phase,
        'error_rms': float(error_rms),
        'fundamental_amplitude': float(fundamental_amplitude),
        'dc_offset': float(dc_offset),
        'normalized_freq': float(normalized_freq),
    }
