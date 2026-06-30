"""Core computation for value-binned error analysis."""

import numpy as np
from typing import Any
from adctoolbox.fundamentals.fit_sine_4param import fit_sine_4param
from adctoolbox.aout._fit_diagnostics import extract_fit_diagnostics
from adctoolbox.aout._infer_signal_range import _infer_signal_range

def rearrange_error_by_value(
    signal: np.ndarray,
    norm_freq: float = None,
    n_bins: int = 100,
    clip_percent: float = 0.01,
    value_range: tuple[float, float | None] = None,
    max_iterations: int = 1,
    tolerance: float = 1e-9,
    return_fit: bool = False,
) -> dict[str, Any]:
    """
    Compute value-binned residual metrics.
    Maps input signal linearly to [0, n_bins-1].
    Fit controls are forwarded to fit_sine_4param.
    """

    signal = np.asarray(signal).flatten()

    if signal.size == 0:
        raise ValueError("signal must not be empty")
    if not np.all(np.isfinite(signal)):
        raise ValueError("signal must contain only finite values")

    if n_bins is None:
        n_bins = 100
    if not isinstance(n_bins, (int, np.integer)) or n_bins <= 0:
        raise ValueError("n_bins must be a positive integer")
    n_bins = int(n_bins)

    if not np.isfinite(clip_percent) or clip_percent < 0:
        raise ValueError("clip_percent must be a finite non-negative value")
    if value_range is not None:
        if len(value_range) != 2:
            raise ValueError("value_range must define finite increasing min/max values")
        range_min, range_max = value_range
        if not np.isfinite(range_min) or not np.isfinite(range_max) or range_min >= range_max:
            raise ValueError("value_range must define finite increasing min/max values")

    # Determine boundaries
    v_min, v_max = _infer_signal_range(signal, value_range)
    if not np.isfinite(v_min) or not np.isfinite(v_max) or v_min >= v_max:
        raise ValueError("value_range must define finite increasing min/max values")

    # Map physical values to bin indices [0, n_bins-1]
    scale_range = v_max - v_min
    if scale_range == 0: 
        scale = 0.0
    else:
        scale = (n_bins - 1) / scale_range

    raw_indices = (signal - v_min) * scale
    bin_indices = np.round(raw_indices).astype(int)
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    value_bin_centers = np.linspace(v_min, v_max, n_bins)

    # Fit sine wave and compute residuals
    fit_kwargs = {"max_iterations": max_iterations, "tolerance": tolerance}
    if norm_freq is None or np.isnan(norm_freq):
        fit_res = fit_sine_4param(signal, **fit_kwargs)
        norm_freq = fit_res['frequency']
    else:        
        fit_res = fit_sine_4param(signal, frequency_estimate=norm_freq, **fit_kwargs)

    fitted_signal = fit_res['fitted_signal']
    error = signal - fitted_signal

    # Filter out edges if requested
    if clip_percent > 0:
        margin = int(n_bins * clip_percent)
        if margin * 2 >= n_bins:
            raise ValueError("clip_percent removes all effective bins")
        valid_mask = (bin_indices >= margin) & (bin_indices <= n_bins - 1 - margin)
    else:
        valid_mask = np.ones(len(bin_indices), dtype=bool)

    if not np.any(valid_mask):
        raise ValueError("clip_percent removes all effective samples")

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

    result = {
        'error_mean': error_mean,
        'error_rms': error_rms,
        'bin_centers': np.arange(n_bins),
        'value_bin_centers': value_bin_centers,
        'count_per_bin': count_per_bin,
        'bin_indices': bin_indices,
        'error': error,
        'fitted_signal': fitted_signal,
        'n_bins': n_bins,
        'norm_freq': float(norm_freq),
    }
    if return_fit:
        result['fit'] = extract_fit_diagnostics(fit_res)

    return result
