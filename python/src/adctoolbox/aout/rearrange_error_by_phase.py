"""Rearrange error by phase using AM/PM separation (core computation kernel).

Core computation kernel for computing phase error metrics by decomposing
residual error into amplitude modulation (AM) and phase modulation (PM)
components. Supports multiple fitting modes for different analysis needs.

Consolidates functionality from compute_phase_error_from_raw and
compute_phase_error_from_binned into a single unified interface.
"""

import numpy as np
from typing import Dict, Optional
from .fit_sine_harmonics import fit_sine_harmonics
from .fit_error_phase import fit_error_phase


def rearrange_error_by_phase(
    signal: np.ndarray,
    normalized_freq: float,
    mode: str = "raw",
    bin_count: int = 100
) -> Dict[str, np.ndarray]:
    """Rearrange error by phase using AM/PM separation.

    Computes phase error metrics by decomposing residual error into amplitude
    modulation (AM) and phase modulation (PM) components. Supports multiple
    fitting modes for flexibility in analysis.

    Parameters
    ----------
    signal : np.ndarray
        Input signal, 1D numpy array.
    normalized_freq : float
        Normalized frequency (f/fs), range 0-0.5.
    mode : str, default="raw"
        Fitting mode for AM/PM separation:
        - "raw": Fit directly to all error samples (high precision, sensitive to outliers)
        - "binned": Fit to binned RMS values (robust to outliers, trend analysis)
    bin_count : int, default=100
        Number of phase bins (only used if mode="binned")

    Returns
    -------
    dict : Dictionary containing:

        **Common outputs (both modes):**
        - 'am_param': float
            Amplitude modulation parameter (amplitude noise)
        - 'pm_param': float
            Phase modulation parameter (phase jitter in radians)
        - 'baseline': float
            Baseline noise floor (constant error variance)
        - 'fitted_signal': np.ndarray
            Reconstructed fundamental signal from fitting
        - 'error': np.ndarray
            Residual error (signal - fitted_signal)
        - 'phase': np.ndarray
            Phase values for each sample in radians
        - 'fundamental_amplitude': float
            Amplitude of fitted fundamental component
        - 'dc_offset': float
            DC offset of fitted signal
        - 'normalized_freq': float
            Input normalized frequency (echoed for convenience)

        **Raw mode only:**
        - 'error_rms': float
            Total RMS error across all samples

        **Binned mode only:**
        - 'erms': np.ndarray
            RMS error per phase bin
        - 'emean': np.ndarray
            Mean error per phase bin
        - 'phase_bins': np.ndarray
            Phase bin centers in radians
        - 'bin_counts': np.ndarray
            Number of samples in each phase bin

    Raises
    ------
    ValueError
        If normalized_freq is not in range (0, 0.5) or mode is invalid

    Notes
    -----
    Both modes fit the AM/PM model to residual error data:

        error²(φ) = am² * cos²(φ) + pm² * A² * sin²(φ) + baseline

    Where:
    - am: Amplitude noise (constant across phase)
    - pm: Phase jitter in radians (varies with phase)
    - A: Fundamental amplitude
    - baseline: Constant noise floor

    **Mode Comparison:**

    Raw mode:
    - Uses all error samples for fitting
    - High precision estimates
    - Sensitive to outliers
    - Lower computational cost
    - Best for clean signals

    Binned mode:
    - Bins error by phase, fits RMS values
    - Robust to outliers
    - Smooth trend estimates
    - Better for trend analysis
    - Best for noisy signals with systematic variation

    Examples
    --------
    >>> import numpy as np
    >>> from adctoolbox.aout import rearrange_error_by_phase

    >>> # High-precision analysis (raw mode)
    >>> sig = np.sin(2*np.pi*0.1*np.arange(1000)) + 0.01*np.random.randn(1000)
    >>> result_raw = rearrange_error_by_phase(sig, normalized_freq=0.1, mode="raw")
    >>> print(f"AM (raw): {result_raw['am_param']:.3e}, PM (raw): {result_raw['pm_param']:.3e}")

    >>> # Robust trend analysis (binned mode)
    >>> result_binned = rearrange_error_by_phase(sig, normalized_freq=0.1, mode="binned", bin_count=100)
    >>> print(f"AM (binned): {result_binned['am_param']:.3e}, PM (binned): {result_binned['pm_param']:.3e}")
    """

    # Input validation
    signal = np.asarray(signal).flatten()
    n_samples = len(signal)

    if not (0 < normalized_freq < 0.5):
        raise ValueError(f"normalized_freq must be in range (0, 0.5), got {normalized_freq}")
    if mode not in ("raw", "binned"):
        raise ValueError(f"mode must be 'raw' or 'binned', got {mode}")

    # Step 1: Fit fundamental sinewave (DC + cos + sin)
    W, fitted_signal, basis_matrix, phase = fit_sine_harmonics(
        signal, freq=normalized_freq, order=1, include_dc=True
    )

    # Extract DC and fundamental amplitude
    dc_offset = W[0]
    cos_coeff = W[1]
    sin_coeff = W[2]
    fundamental_amplitude = np.sqrt(cos_coeff**2 + sin_coeff**2)

    # Step 2: Compute residual error
    error = signal - fitted_signal

    # Step 3: Perform AM/PM separation using fit_error_phase kernel
    phase_error_result = fit_error_phase(
        error=error,
        phase=phase,
        fundamental_amplitude=fundamental_amplitude,
        mode=mode,  # Pass through the mode parameter
        include_baseline=True,  # Always include baseline estimation
        bin_count=bin_count if mode == "binned" else None
    )

    am_param = phase_error_result['am_param']
    pm_param = phase_error_result['pm_param']
    baseline = phase_error_result['baseline']

    # Build common results
    results = {
        'am_param': float(am_param),
        'pm_param': float(pm_param),
        'baseline': float(baseline),
        'fitted_signal': fitted_signal,
        'error': error,
        'phase': phase,
        'fundamental_amplitude': float(fundamental_amplitude),
        'dc_offset': float(dc_offset),
        'normalized_freq': float(normalized_freq),
    }

    # Add mode-specific outputs
    if mode == "raw":
        # Raw mode: compute error RMS
        error_rms = np.sqrt(np.mean(error**2))
        results['error_rms'] = float(error_rms)

    else:  # mode == "binned"
        # Binned mode: compute phase binning statistics
        phase_bins = np.linspace(0, 2*np.pi, bin_count, endpoint=False)
        bin_width = 2 * np.pi / bin_count

        # Initialize binning arrays
        bin_counts_arr = np.zeros(bin_count)
        error_sum = np.zeros(bin_count)
        error_sq_sum = np.zeros(bin_count)

        # Assign each sample to a phase bin
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
        with np.errstate(divide='ignore', invalid='ignore'):
            emean = np.where(bin_counts_arr > 0, error_sum / bin_counts_arr, np.nan)
            erms = np.where(
                bin_counts_arr > 0,
                np.sqrt(error_sq_sum / bin_counts_arr),
                np.nan
            )

        results['erms'] = erms
        results['emean'] = emean
        results['phase_bins'] = phase_bins
        results['bin_counts'] = bin_counts_arr

    return results
