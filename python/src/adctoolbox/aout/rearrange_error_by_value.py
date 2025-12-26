"""Core computation for value-binned error analysis."""

import numpy as np
from typing import Any
from adctoolbox.fundamentals.fit_sine_4param import fit_sine_4param
from adctoolbox.aout._infer_signal_range import _infer_signal_range

def rearrange_error_by_value(
    signal: np.ndarray,
    norm_freq: float = None,
    n_bins: int = 100,
    clip_percent: float = 0.01,
    value_range: tuple[float, float | None] = None,
) -> dict[str, Any]:
    """
    Compute value-binned error metrics.
    Maps input signal linearly to [0, n_bins-1].
    """
    
    signal = np.asarray(signal).flatten()
    
    if n_bins is None: 
        n_bins = 100

    # Determine boundaries
    v_min, v_max = _infer_signal_range(signal, value_range)

    # Map physical values to bin indices [0, n_bins-1]
    scale_range = v_max - v_min
    if scale_range == 0: 
        scale = 0.0
    else:
        scale = (n_bins - 1) / scale_range

    raw_indices = (signal - v_min) * scale
    bin_indices = np.round(raw_indices).astype(int)
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Fit sine wave and compute residuals
    if norm_freq is None or np.isnan(norm_freq):
        fit_res = fit_sine_4param(signal)
        norm_freq = fit_res['frequency']
    else:        
        fit_res = fit_sine_4param(signal, frequency_estimate=norm_freq)

    fitted_signal = fit_res['fitted_signal']
    error = signal - fitted_signal

    # Filter out edges if requested
    if clip_percent > 0:
        margin = int(n_bins * clip_percent)
        valid_mask = (bin_indices >= margin) & (bin_indices <= n_bins - 1 - margin)
    else:
        valid_mask = np.ones(len(bin_indices), dtype=bool)

    valid_indices = bin_indices[valid_mask]
    valid_error = error[valid_mask]

    # Compute statistics using vectorized operations
    count_per_bin = np.bincount(valid_indices, minlength=n_bins)
    sum_err_per_bin = np.bincount(valid_indices, weights=valid_error, minlength=n_bins)
    sum_sq_err_per_bin = np.bincount(valid_indices, weights=valid_error**2, minlength=n_bins)

    with np.errstate(divide='ignore', invalid='ignore'):
        # INL Profile
        error_mean = np.where(count_per_bin > 0, sum_err_per_bin / count_per_bin, np.nan)
        # Noise Profile
        error_rms = np.where(count_per_bin > 0, np.sqrt(sum_sq_err_per_bin / count_per_bin), np.nan)

    return {
        'error_mean': error_mean,
        'error_rms': error_rms,
        'bin_centers': np.arange(n_bins),
        'bin_indices': bin_indices,
        'error': error,
        'fitted_signal': fitted_signal,
        'n_bins': n_bins,
        'norm_freq': float(norm_freq),
    }