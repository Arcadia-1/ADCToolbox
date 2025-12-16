"""Fit phase error using AM/PM separation (core math kernel).

Core mathematical kernel for decomposing residual error into amplitude modulation
(AM) and phase modulation (PM) components using least-squares fitting.

Supports multiple fitting modes:
- raw: Fit AM/PM model directly to all error samples (high precision)
- binned: Fit AM/PM model to binned RMS values (robust to outliers)

Supports baseline noise estimation:
- include_baseline=True: Estimates constant noise floor (recommended)
- include_baseline=False: Assumes zero baseline (forced to only AM/PM terms)
"""

import numpy as np
from typing import Dict, Tuple, Optional


def fit_error_phase(
    error: np.ndarray,
    phase: np.ndarray,
    fundamental_amplitude: float,
    mode: str = "raw",
    include_baseline: bool = True,
    bin_count: Optional[int] = 100
) -> Dict[str, float]:
    """Fit phase error using AM/PM separation.

    Decomposes residual error into amplitude modulation (AM) and phase modulation (PM)
    components using least-squares fitting. Supports multiple modes and baseline estimation.

    Parameters
    ----------
    error : np.ndarray
        Residual error signal, shape (N,)
    phase : np.ndarray
        Phase vector in radians, shape (N,)
    fundamental_amplitude : float
        Amplitude of the fundamental component (for PM normalization)
    mode : str, default="raw"
        Fitting mode:
        - "raw": Fit directly to all error samples (high precision, sensitive to outliers)
        - "binned": Fit to binned RMS values (robust, requires bin_count parameter)
    include_baseline : bool, default=True
        Include baseline (constant noise floor) in the fitting:
        - True: Model = am² * cos²(φ) + pm² * A² * sin²(φ) + baseline
        - False: Model = am² * cos²(φ) + pm² * A² * sin²(φ)
    bin_count : int, optional
        Number of phase bins (only used if mode="binned", default: 100)

    Returns
    -------
    dict
        Dictionary containing:
        - 'am_param': float
            Amplitude modulation parameter (amplitude noise)
        - 'pm_param': float
            Phase modulation parameter (phase jitter in radians)
        - 'baseline': float
            Baseline noise floor (0.0 if include_baseline=False)
        - 'fitting_error': float
            RMS fitting residual

    Notes
    -----
    AM/PM Model:
        error²(φ) = am² * cos²(φ) + pm² * A² * sin²(φ) + baseline

    Where:
    - am: Amplitude noise (constant across phase)
    - pm: Phase jitter in radians (varies with phase)
    - A: Fundamental amplitude
    - baseline: Constant noise floor (optional)

    Examples
    --------
    >>> import numpy as np
    >>> N = 1000
    >>> phase = 2 * np.pi * np.arange(N) / N
    >>> error = 0.01 * np.cos(phase) + 0.001 * np.random.randn(N)
    >>> result = fit_error_phase(error, phase, fundamental_amplitude=0.5, mode="raw")
    >>> print(f"AM: {result['am_param']:.3f}, PM: {result['pm_param']:.3f}")
    """

    # Input validation
    error = np.asarray(error).flatten()
    phase = np.asarray(phase).flatten()
    n_samples = len(error)

    if len(phase) != n_samples:
        raise ValueError(f"error and phase must have same length, got {n_samples} and {len(phase)}")
    if fundamental_amplitude <= 0:
        raise ValueError(f"fundamental_amplitude must be positive, got {fundamental_amplitude}")
    if mode not in ("raw", "binned"):
        raise ValueError(f"mode must be 'raw' or 'binned', got {mode}")

    if mode == "raw":
        return _fit_error_phase_raw(error, phase, fundamental_amplitude, include_baseline)
    else:  # mode == "binned"
        if bin_count is None:
            bin_count = 100
        return _fit_error_phase_binned(error, phase, fundamental_amplitude, include_baseline, bin_count)


def _fit_error_phase_raw(
    error: np.ndarray,
    phase: np.ndarray,
    fundamental_amplitude: float,
    include_baseline: bool
) -> Dict[str, float]:
    """Fit AM/PM model directly to raw error samples."""

    n_samples = len(error)

    # Build sensitivity curves for each sample
    phase_wrapped = np.mod(phase, 2 * np.pi)
    am_sensitivity = np.cos(phase_wrapped) ** 2  # Amplitude modulation sensitivity
    pm_sensitivity = np.sin(phase_wrapped) ** 2  # Phase modulation sensitivity

    # Square the error for fitting
    error_squared = error ** 2

    # Build design matrix
    if include_baseline:
        # Model: error² = am² * cos²(φ) + pm²*A² * sin²(φ) + baseline
        A_matrix = np.column_stack([am_sensitivity, pm_sensitivity, np.ones(n_samples)])
    else:
        # Model: error² = am² * cos²(φ) + pm²*A² * sin²(φ)
        A_matrix = np.column_stack([am_sensitivity, pm_sensitivity])

    # Solve least-squares problem
    try:
        params, residuals, rank, s = np.linalg.lstsq(A_matrix, error_squared, rcond=None)
    except np.linalg.LinAlgError:
        # Fallback to zeros if fitting fails
        if include_baseline:
            params = np.array([0.0, 0.0, np.mean(error_squared)])
        else:
            params = np.array([0.0, 0.0])

    # Extract parameters with validation
    am_param_sq = max(0.0, params[0])
    pm_param_sq = max(0.0, params[1])
    baseline = max(0.0, params[2]) if include_baseline else 0.0

    am_param = np.sqrt(am_param_sq)

    # PM parameter: sqrt(pm²*A²) / A = pm (in radians)
    if fundamental_amplitude > 1e-10:
        pm_param = np.sqrt(pm_param_sq) / fundamental_amplitude
    else:
        pm_param = 0.0

    # Calculate fitting residual
    fitted_error_sq = A_matrix @ params
    fitting_residual = np.sqrt(np.mean((error_squared - fitted_error_sq) ** 2))

    return {
        "am_param": float(am_param),
        "pm_param": float(pm_param),
        "baseline": float(baseline),
        "fitting_error": float(fitting_residual),
    }


def _fit_error_phase_binned(
    error: np.ndarray,
    phase: np.ndarray,
    fundamental_amplitude: float,
    include_baseline: bool,
    bin_count: int
) -> Dict[str, float]:
    """Fit AM/PM model to binned RMS error values."""

    n_samples = len(error)

    # Bin error by phase
    phase_bins = np.linspace(0, 2 * np.pi, bin_count, endpoint=False)
    bin_width = 2 * np.pi / bin_count

    # Initialize binning arrays
    bin_counts_arr = np.zeros(bin_count)
    error_sq_sum = np.zeros(bin_count)

    # Assign each sample to a phase bin
    phase_wrapped = np.mod(phase, 2 * np.pi)
    bin_indices = np.floor(phase_wrapped / bin_width).astype(int)
    bin_indices = np.clip(bin_indices, 0, bin_count - 1)

    # Accumulate statistics per bin
    for i in range(n_samples):
        bin_idx = bin_indices[i]
        bin_counts_arr[bin_idx] += 1
        error_sq_sum[bin_idx] += error[i] ** 2

    # Compute RMS per bin
    with np.errstate(divide="ignore", invalid="ignore"):
        erms = np.where(
            bin_counts_arr > 0, np.sqrt(error_sq_sum / bin_counts_arr), np.nan
        )
    erms_squared = erms ** 2

    # Build sensitivity curves
    am_sensitivity = np.cos(phase_bins) ** 2  # Amplitude modulation sensitivity
    pm_sensitivity = np.sin(phase_bins) ** 2  # Phase modulation sensitivity

    # Filter out NaN bins (empty bins)
    valid_mask = ~np.isnan(erms_squared)
    if np.sum(valid_mask) < 2:
        # Not enough valid bins for fitting
        return {
            "am_param": 0.0,
            "pm_param": 0.0,
            "baseline": float(np.nanmean(erms_squared)) if np.any(valid_mask) else 0.0,
            "fitting_error": 0.0,
        }

    erms_squared_valid = erms_squared[valid_mask]

    # Build design matrix
    if include_baseline:
        # Model: RMS² = am² * cos²(φ) + pm²*A² * sin²(φ) + baseline
        A_matrix = np.column_stack(
            [
                am_sensitivity[valid_mask],
                pm_sensitivity[valid_mask],
                np.ones(np.sum(valid_mask)),
            ]
        )
    else:
        # Model: RMS² = am² * cos²(φ) + pm²*A² * sin²(φ)
        A_matrix = np.column_stack([am_sensitivity[valid_mask], pm_sensitivity[valid_mask]])

    # Solve least-squares problem
    try:
        params, residuals, rank, s = np.linalg.lstsq(A_matrix, erms_squared_valid, rcond=None)
    except np.linalg.LinAlgError:
        # Fallback to zeros if fitting fails
        if include_baseline:
            params = np.array([0.0, 0.0, np.nanmean(erms_squared_valid)])
        else:
            params = np.array([0.0, 0.0])

    # Extract parameters with validation
    am_param_sq = max(0.0, params[0])
    pm_param_sq = max(0.0, params[1])
    baseline = max(0.0, params[2]) if include_baseline else 0.0

    am_param = np.sqrt(am_param_sq)

    # PM parameter: sqrt(pm²*A²) / A = pm (in radians)
    if fundamental_amplitude > 1e-10:
        pm_param = np.sqrt(pm_param_sq) / fundamental_amplitude
    else:
        pm_param = 0.0

    # Calculate fitting residual
    fitted_erms_sq = A_matrix @ params
    fitting_residual = np.sqrt(np.mean((erms_squared_valid - fitted_erms_sq) ** 2))

    return {
        "am_param": float(am_param),
        "pm_param": float(pm_param),
        "baseline": float(baseline),
        "fitting_error": float(fitting_residual),
    }
