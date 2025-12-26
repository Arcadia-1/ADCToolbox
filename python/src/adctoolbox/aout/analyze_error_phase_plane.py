"""Error phase plane analysis: visualize harmonic distortion by removing fundamental."""

import numpy as np
import matplotlib.pyplot as plt
from adctoolbox.fundamentals import fit_sine_4param

def analyze_error_phase_plane(data, fs=1.0, ax=None, title=None, create_plot: bool = True,
                               fit_polynomial_order=3, detect_hysteresis=True):
    """
    Residual Phase Plane: Magnify harmonic distortion by removing the fundamental.

    This technique is 1000x more sensitive to harmonics than regular phase planes.
    By fitting and subtracting the ideal sine wave, tiny harmonic distortions become visible.

    Parameters
    ----------
    data : array_like
        1D array of ADC codes or voltage data (should be sine wave).
    fs : float
        Sampling frequency (used only for display text).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    title : str, optional
        Custom title for the plot.
    create_plot : bool
        Whether to render the plot immediately.
    fit_polynomial_order : int
        Order of polynomial to fit the residual trend (default 3 for HD2/HD3).
    detect_hysteresis : bool
        If True, separately fit rising and falling edges to detect memory effects.

    Returns
    -------
    dict
        'residual': array of error values (data - fitted_sine)
        'fitted_params': dict from IEEE fit_sine_4param
        'trend_coeffs': polynomial coefficients of the trend line
        'hysteresis_gap': average gap between rising/falling trends (if detected)

    Notes
    -----
    Units are automatically detected:
    - Data range < 50: Assumed voltage, displayed in µV
    - Data range ≥ 50: Assumed ADC codes, displayed in LSB

    What to look for in the plot:
    - Parabola (U-shape): HD2 (2nd harmonic) - gain asymmetry
    - S-curve: HD3 (3rd harmonic) - compression/saturation
    - Hysteresis loop (rising≠falling): Memory effect, reference settling issues
    - Horizontal line: Good linearity (only random noise)
    - Sharp breaks at edges: Clipping
    - Thick scatter: High noise (poor SNR)
    """
    data = np.asarray(data).flatten()
    N = len(data)

    # --- 1. Auto-detect unit and scale factor ---
    # Heuristic: voltage data is typically < 50, ADC codes are > 100
    is_voltage = np.max(np.abs(data)) < 50
    scale_factor = 1e6 if is_voltage else 1.0
    unit_str = 'uV' if is_voltage else 'LSB'

    # --- 2. IEEE 1057 Sine Fit (Fast & Robust) ---
    # Use standardized 4-parameter fit instead of curve_fit
    fit_result = fit_sine_4param(data, max_iterations=3)
    residual = fit_result['residuals']

    # --- 3. Robust Polynomial Fitting (Exclude Clipping) ---
    # Only fit central 96% of amplitude range to avoid clipping artifacts
    sort_idx = np.argsort(data)
    x_sorted = data[sort_idx]
    residual_sorted = residual[sort_idx]

    # Exclude top/bottom 2% to handle clipping
    valid_mask = (data > np.percentile(data, 2)) & (data < np.percentile(data, 98))

    # Fit polynomial on non-clipped region
    poly_coeff = np.polyfit(data[valid_mask], residual[valid_mask], fit_polynomial_order)
    poly_trend = np.polyval(poly_coeff, x_sorted)

    # --- 4. Hysteresis Detection (Memory Effect) ---
    hysteresis_gap = 0.0
    coeff_rise = None
    coeff_fall = None

    if detect_hysteresis:
        # Calculate gradient to separate rising from falling edges
        gradient = np.gradient(data)
        mask_rise = gradient > 0
        mask_fall = gradient < 0

        # Only compute if we have sufficient samples in each direction
        if np.sum(mask_rise) > 100 and np.sum(mask_fall) > 100:
            # Fit rising edge
            valid_rise = mask_rise & valid_mask
            if np.sum(valid_rise) > 50:
                coeff_rise = np.polyfit(data[valid_rise], residual[valid_rise], fit_polynomial_order)

            # Fit falling edge
            valid_fall = mask_fall & valid_mask
            if np.sum(valid_fall) > 50:
                coeff_fall = np.polyfit(data[valid_fall], residual[valid_fall], fit_polynomial_order)

            # Calculate hysteresis gap (average separation)
            if coeff_rise is not None and coeff_fall is not None:
                trend_rise = np.polyval(coeff_rise, x_sorted)
                trend_fall = np.polyval(coeff_fall, x_sorted)
                hysteresis_gap = np.mean(np.abs(trend_rise - trend_fall))

    # --- 5. Visualization ---
    if create_plot or ax is not None:
        # Use current axes if available, otherwise use pyplot's implicit axes
        if ax is None:
            ax = plt.gca()

        # Apply scale factor for display
        residual_display = residual * scale_factor
        poly_trend_display = poly_trend * scale_factor

        ax.scatter(data, residual_display, s=0.5, c='gray', alpha=0.3)

        # Hysteresis visualization: separate rising/falling trends
        if coeff_rise is not None and coeff_fall is not None:
            trend_rise_display = np.polyval(coeff_rise, x_sorted) * scale_factor
            trend_fall_display = np.polyval(coeff_fall, x_sorted) * scale_factor
            ax.plot(x_sorted, trend_rise_display, 'g-', lw=4, label='Rising (Memory)')
            ax.plot(x_sorted, trend_fall_display, 'm-', lw=4, label='Falling (Memory)')
        else:
            # Standard trend line if no hysteresis
            ax.plot(x_sorted, poly_trend_display, 'r-', lw=4, label='Trend')

        # Formatting
        if title is not None:
            ax.set_title(title, fontsize=12, fontweight='bold')
        else:
            gap_str = f", Hysteresis: {hysteresis_gap*scale_factor:.1f} {unit_str}" if hysteresis_gap > 0 else ""
            ax.set_title(f"Error Phase Plane{gap_str}", fontsize=12, fontweight='bold')

        ax.set_xlabel("Signal Amplitude (V)" if unit_str == 'uV' else "Signal Amplitude (Code)", fontsize=10)
        ax.set_ylabel(f"Error / Residual ({unit_str})", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Show statistics
        residual_display_std = np.std(residual_display)
        residual_display_peak = np.max(np.abs(residual_display))

        stats_text = f"RMS: {residual_display_std:.1f} {unit_str}\nPeak: {residual_display_peak:.1f} {unit_str}"
        if hysteresis_gap > 0:
            stats_text += f"\nHyst: {hysteresis_gap*scale_factor:.1f} {unit_str}"

        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right',
               fontsize=9,
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    return {
        'residual': residual,
        'fitted_params': fit_result,
        'trend_coeffs': poly_coeff,
        'hysteresis_gap': hysteresis_gap
    }
