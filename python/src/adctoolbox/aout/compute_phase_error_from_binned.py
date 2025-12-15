"""
Compute phase error using binned data approach (trend analysis).

Core computation kernel for computing phase error metrics by binning error
data by phase. This approach is useful for trend analysis and noise separation.
"""

import numpy as np
from typing import Dict
from .decompose_harmonics import fit_sinewave_components


def compute_phase_error_from_binned(
    signal: np.ndarray,
    normalized_freq: float,
    bin_count: int = 100
) -> Dict[str, np.ndarray]:
    """
    Compute phase error using binned data approach (trend analysis).

    This function fits a fundamental sinewave to the input signal, computes
    the residual error, bins it by phase, and performs AM/PM separation
    using least-squares fitting on the binned data.

    Parameters
    ----------
    signal : np.ndarray
        Input signal, 1D numpy array.
    normalized_freq : float
        Normalized frequency (f/fs), range 0-0.5.
    bin_count : int, default=100
        Number of phase bins for binning error data (typically 50-200).

    Returns
    -------
    dict : Dictionary containing:
        - 'erms' : np.ndarray
            RMS error per phase bin, shape (bin_count,).
        - 'emean' : np.ndarray
            Mean error per phase bin, shape (bin_count,).
        - 'phase_bins' : np.ndarray
            Phase bin centers in radians, shape (bin_count,).
        - 'am_param' : float
            Amplitude modulation parameter (amplitude noise).
        - 'pm_param' : float
            Phase modulation parameter (phase noise in radians).
        - 'baseline' : float
            Baseline noise floor.
        - 'fitted_signal' : np.ndarray
            Reconstructed fundamental signal from fitting.
        - 'error' : np.ndarray
            Residual error (signal - fitted_signal).
        - 'phase' : np.ndarray
            Phase values for each sample in radians.
        - 'bin_counts' : np.ndarray
            Number of samples in each phase bin.

    Notes
    -----
    The binned approach computes error statistics (mean and RMS) within each
    phase bin, then fits AM/PM models to the binned RMS values:

        RMS²(φ) = am_param² * cos²(φ) + pm_param² * sin²(φ) + baseline

    This approach is robust to outliers and provides smooth trend estimates.

    Examples
    --------
    >>> sig = np.sin(2*np.pi*0.1*np.arange(1000)) + 0.01*np.random.randn(1000)
    >>> result = compute_phase_error_from_binned(sig, normalized_freq=0.1)
    >>> print(f"AM param: {result['am_param']:.3e}")
    >>> print(f"PM param: {result['pm_param']:.3e}")
    """
    # Validate inputs
    signal = np.asarray(signal).flatten()
    n_samples = len(signal)

    if not (0 < normalized_freq < 0.5):
        raise ValueError(f"normalized_freq must be in range (0, 0.5), got {normalized_freq}")
    if bin_count < 10:
        raise ValueError(f"bin_count must be >= 10, got {bin_count}")

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

    # Step 3: Bin error by phase
    phase_bins = np.linspace(0, 2*np.pi, bin_count, endpoint=False)
    bin_width = 2 * np.pi / bin_count

    # Initialize binning arrays
    bin_counts_arr = np.zeros(bin_count)
    error_sum = np.zeros(bin_count)
    error_sq_sum = np.zeros(bin_count)

    # Assign each sample to a phase bin
    # Map phase to [0, 2π) and determine bin index
    phase_wrapped = np.mod(phase, 2*np.pi)
    bin_indices = np.floor(phase_wrapped / bin_width).astype(int)
    bin_indices = np.clip(bin_indices, 0, bin_count - 1)

    # Accumulate statistics per bin
    for i in range(n_samples):
        bin_idx = bin_indices[i]
        bin_counts_arr[bin_idx] += 1
        error_sum[bin_idx] += error[i]
        error_sq_sum[bin_idx] += error[i]**2

    # Compute mean and RMS per bin
    # Use np.where to avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        emean = np.where(bin_counts_arr > 0, error_sum / bin_counts_arr, np.nan)
        erms = np.where(
            bin_counts_arr > 0,
            np.sqrt(error_sq_sum / bin_counts_arr),
            np.nan
        )

    # Step 4: Perform AM/PM separation using least-squares on binned RMS
    # Model: RMS²(φ) = am² * cos²(φ) + pm² * A² * sin²(φ) + baseline
    # Where A is the fundamental amplitude

    # Build sensitivity curves
    am_sensitivity = np.cos(phase_bins)**2  # Amplitude modulation sensitivity
    pm_sensitivity = np.sin(phase_bins)**2  # Phase modulation sensitivity

    # Filter out NaN bins (empty bins)
    valid_mask = ~np.isnan(erms)
    if np.sum(valid_mask) < 3:
        # Not enough valid bins for fitting
        am_param = 0.0
        pm_param = 0.0
        baseline = np.nanmean(erms**2) if np.any(valid_mask) else 0.0
    else:
        erms_squared = erms[valid_mask]**2

        # Design matrix: [AM_sensitivity, PM_sensitivity, ones]
        A_matrix = np.column_stack([
            am_sensitivity[valid_mask],
            pm_sensitivity[valid_mask],
            np.ones(np.sum(valid_mask))
        ])

        # Solve least-squares: A @ [am², pm²*A², baseline] ≈ RMS²
        try:
            params, residuals, rank, s = np.linalg.lstsq(A_matrix, erms_squared, rcond=None)
        except np.linalg.LinAlgError:
            # Fallback to zeros if fitting fails
            params = np.array([0.0, 0.0, np.nanmean(erms_squared)])

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
        'erms': erms,
        'emean': emean,
        'phase_bins': phase_bins,
        'am_param': float(am_param),
        'pm_param': float(pm_param),
        'baseline': float(baseline),
        'fitted_signal': fitted_signal,
        'error': error,
        'phase': phase,
        'bin_counts': bin_counts_arr,
        'fundamental_amplitude': float(fundamental_amplitude),
        'dc_offset': float(dc_offset),
        'normalized_freq': float(normalized_freq),
    }
