"""Rearrange error by phase using AM/PM separation (driver layer).

Dual-track parallel design:
- Path A (Raw): Fit all N samples → highest precision AM/PM values + r_squared_raw
- Path B (Binned): Compute binned statistics → visualization
- Cross-validation: Use Path A coeffs to predict Path B trend → r_squared_binned
"""

import numpy as np

from adctoolbox.fundamentals.fit_sine_4param import fit_sine_4param
from adctoolbox.aout._fit_error_phase import _fit_error_phase

def rearrange_error_by_phase(
    signal: np.ndarray,
    norm_freq: float = None,
    n_bins: int = 100,
    include_base_noise: bool = True
) -> dict[str, np.ndarray]:
    """Rearrange error by phase using dual-track AM/PM separation.

    Parameters
    ----------
    signal : np.ndarray
        Input signal, 1D numpy array.
    norm_freq : float, optional
        Normalized frequency (f/fs), range 0-0.5. If None, auto-detected via FFT.
    n_bins : int, default=100
        Number of phase bins for visualization.
    include_base_noise : bool, default=True
        Include base noise floor in fitting model.

    Returns
    -------
    dict
        Numerics (from raw fitting):
            am_noise_rms_v, pm_noise_rms_v, pm_noise_rms_rad,
            noise_floor_rms_v, total_rms_v
        Validation (dual R²):
            r_squared_raw: AM/PM energy ratio in total noise (typically low, ~0.05)
            r_squared_binned: Model fit quality on binned trend (typically high, ~0.95)
        Visualization (binned):
            bin_error_rms_v, bin_error_mean_v, phase_bin_centers_rad, bin_counts
        Metadata:
            amplitude, dc_offset, norm_freq, fitted_signal, error, phase
    """
    signal = np.asarray(signal).flatten()
    n_samples = len(signal)

    # ===== Step 1: Fit fundamental sinewave (Cosine basis) =====
    # fit_sine_4param: y = A*cos(wt) + B*sin(wt) + C
    # phase = atan2(-B, A), so fitted = amplitude * cos(wt + phase)
    fit_result = fit_sine_4param(signal, frequency_estimate=norm_freq, max_iterations=1)
    fitted_signal = fit_result['fitted_signal']
    amplitude = fit_result['amplitude']
    dc_offset = fit_result['dc_offset']

    # Compute phase for each sample: phase = 2*pi*f*t + phase_offset
    t = np.arange(n_samples)
    freq = fit_result['frequency']
    phase_offset = fit_result['phase']
    phase = 2 * np.pi * freq * t + phase_offset

    # Use detected frequency if norm_freq was not provided
    if norm_freq is None:
        norm_freq = freq

    error = signal - fitted_signal
    phase_wrapped = np.mod(phase, 2 * np.pi)

    # ===== Step 2: Path A - Raw fitting (The Numerics) =====
    error_sq = error ** 2
    raw_result = _fit_error_phase(error_sq, phase_wrapped, include_base_noise)
    r_squared_raw = raw_result['r_squared']  # Energy ratio (typically low)
    coeffs = raw_result['coeffs']  # [am², pm², base_noise]

    # --- Use cos(2φ) basis to avoid rank deficiency ---
    # The standard model: y = am²·cos²(φ) + pm²·sin²(φ) + base_noise
    # Is rank-deficient because cos² + sin² = 1
    #
    # Transform to orthogonal basis using: cos²(φ) = (1 + cos(2φ))/2
    #   y = A·cos(2φ) + B
    # where:
    #   A = (am² - pm²) / 2
    #   B = (am² + pm²) / 2 + base_noise
    #
    # Physical interpretation (overlap redistribution built-in):
    #   If A > 0: AM dominates → am² = 2A, pm² = 0, base_noise = B - A
    #   If A < 0: PM dominates → am² = 0, pm² = -2A, base_noise = B + A
    #   If A = 0: Flat → am² = 0, pm² = 0, base_noise = B

    if include_base_noise:
        # Fit y = A·cos(2φ) + B (2 parameters, full rank)
        cos2phi = np.cos(2 * phase_wrapped)
        X = np.column_stack([cos2phi, np.ones_like(cos2phi)])
        coeffs_2param, _, _, _ = np.linalg.lstsq(X, error_sq, rcond=None)
        A, B = coeffs_2param[0], coeffs_2param[1]

        # Physical interpretation
        if A > 0:
            # AM dominates
            am_var_final = max(0.0, 2 * A)
            pm_var_final = 0.0
            base_noise_var = max(0.0, B - A)
        elif A < 0:
            # PM dominates
            am_var_final = 0.0
            pm_var_final = max(0.0, -2 * A)
            base_noise_var = max(0.0, B + A)
        else:
            # Flat (thermal or AM=PM)
            am_var_final = 0.0
            pm_var_final = 0.0
            base_noise_var = max(0.0, B)
    else:
        # No base noise: use original 2-parameter fit (am², pm²)
        am_var_final = max(0.0, coeffs[0])
        pm_var_final = max(0.0, coeffs[1])
        base_noise_var = 0.0

    # Physical interpretation: variances → RMS (peak) values
    am_v = float(np.sqrt(am_var_final))
    pm_v = float(np.sqrt(pm_var_final))

    # Convert PM to radians
    pm_rad = pm_v / amplitude if amplitude > 1e-10 else 0.0

    # ===== Step 3: Path B - Binned statistics (The Visualization) =====
    bin_width = 2 * np.pi / n_bins
    bin_indices = np.floor(phase_wrapped / bin_width).astype(int)
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Vectorized binning using np.bincount
    bin_counts = np.bincount(bin_indices, minlength=n_bins).astype(float)
    bin_sum = np.bincount(bin_indices, weights=error, minlength=n_bins)
    bin_sum_sq = np.bincount(bin_indices, weights=error_sq, minlength=n_bins)

    # Compute mean and RMS per bin
    with np.errstate(divide='ignore', invalid='ignore'):
        bin_error_mean_v = np.where(bin_counts > 0, bin_sum / bin_counts, np.nan)
        bin_error_rms_v = np.where(bin_counts > 0, np.sqrt(bin_sum_sq / bin_counts), np.nan)

    # Phase bin centers: use geometric center (not left edge) for accurate alignment
    phase_bin_centers_rad = np.linspace(0, 2 * np.pi, n_bins, endpoint=False) + bin_width / 2

    # ===== Step 4: Cross-validation (Final coeffs → Path B trend) =====
    # Use final coefficients to measure fit quality (R²)
    valid_mask = bin_counts > 0
    if np.sum(valid_mask) > 2:
        phi_valid = phase_bin_centers_rad[valid_mask]
        rms_sq_actual = bin_error_rms_v[valid_mask] ** 2

        # Predict using final coefficients (Cosine basis)
        am_sen = np.cos(phi_valid) ** 2
        pm_sen = np.sin(phi_valid) ** 2
        rms_sq_pred = am_var_final * am_sen + pm_var_final * pm_sen + base_noise_var

        # Compute R² for binned trend (model confidence)
        ss_res = np.sum((rms_sq_actual - rms_sq_pred) ** 2)
        ss_tot = np.sum((rms_sq_actual - np.mean(rms_sq_actual)) ** 2)

        # For flat data (base_noise-dominated), use normalized RMSE instead
        mean_actual = np.mean(rms_sq_actual)
        if ss_tot > 1e-20 and ss_tot > 0.01 * mean_actual ** 2 * len(rms_sq_actual):
            r_squared_binned = 1.0 - ss_res / ss_tot
        else:
            relative_rmse = np.sqrt(ss_res / len(rms_sq_actual)) / mean_actual if mean_actual > 1e-20 else 1.0
            r_squared_binned = max(0.0, 1.0 - relative_rmse ** 2)
    else:
        r_squared_binned = 0.0

    # ===== Build results =====
    total_rms_v = float(np.sqrt(np.mean(error_sq)))

    return {
        # Numerics (from raw fitting - highest precision)
        'am_noise_rms_v': float(am_v),
        'am_relative': float(am_v / amplitude) if amplitude > 1e-10 else 0.0,
        'pm_noise_rms_v': float(pm_v),
        'pm_noise_rms_rad': float(pm_rad),
        'base_noise_rms_v': float(np.sqrt(base_noise_var)),
        'total_rms_v': total_rms_v,

        # Validation (dual R²)
        'r_squared_raw': float(r_squared_raw),      # Energy ratio (typically low, ~0.05)
        'r_squared_binned': float(r_squared_binned),  # Model confidence (typically high, ~0.95)

        # Visualization (binned)
        'bin_error_rms_v': bin_error_rms_v,
        'bin_error_mean_v': bin_error_mean_v,
        'phase_bin_centers_rad': phase_bin_centers_rad,
        'bin_counts': bin_counts,

        # Metadata
        'amplitude': float(amplitude),
        'dc_offset': float(dc_offset),
        'norm_freq': float(norm_freq),
        'fitted_signal': fitted_signal,
        'error': error,
        'phase': phase,

        # Coefficients for plotting (after smart base_noise detection)
        '_coeffs_raw': coeffs,
        '_coeffs_plot': [am_var_final, pm_var_final, base_noise_var],
        '_include_base_noise': include_base_noise,
    }
