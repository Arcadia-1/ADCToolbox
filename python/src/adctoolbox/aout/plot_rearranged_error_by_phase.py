"""Plot phase error analysis results (visualization layer).

Simplified design:
- Always plots binned bar chart with AM/PM fitted curves
- Displays R² for model validation
- No mode selection needed
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec

def _format_value_with_unit(value_v: float) -> str:
    """Format voltage value with appropriate SI unit prefix."""
    abs_val = abs(value_v)
    if abs_val < 1e-12:  # Treat as zero
        return "0.00 uV"
    elif abs_val >= 1:
        return f"{value_v:.2f} V"
    elif abs_val >= 1e-3:
        return f"{value_v * 1e3:.2f} mV"
    elif abs_val >= 1e-6:
        return f"{value_v * 1e6:.2f} uV"
    elif abs_val >= 1e-9:
        return f"{value_v * 1e9:.2f} nV"
    else:
        return f"{value_v * 1e12:.2f} pV"

def plot_rearranged_error_by_phase(results: dict, axes=None, ax=None, title: str | None = None):
    """Plot phase error analysis results.

    Parameters
    ----------
    results : dict
        Dictionary from rearrange_error_by_phase().
    axes : tuple, optional
        Tuple of (ax1, ax2) for top and bottom panels.
    ax : matplotlib.axes.Axes, optional
        Single axis to split into 2 panels.
    title : str, optional
        Test setup description for title.
    """
    # Extract data
    error = results.get('error', np.array([]))
    phase = results.get('phase', np.array([]))
    fitted_signal = results.get('fitted_signal', np.array([]))

    bin_error_rms_v = results.get('bin_error_rms_v', np.array([]))
    bin_error_mean_v = results.get('bin_error_mean_v', np.array([]))
    phase_bin_centers_rad = results.get('phase_bin_centers_rad', np.array([]))

    am_noise_rms_v = results.get('am_noise_rms_v', 0.0)
    pm_noise_rms_v = results.get('pm_noise_rms_v', 0.0)
    pm_noise_rms_rad = results.get('pm_noise_rms_rad', 0.0)
    base_noise_rms_v = results.get('base_noise_rms_v', 0.0)
    r_squared_binned = results.get('r_squared_binned', 0.0)  # Model confidence
    include_base_noise = results.get('_include_base_noise', True)

    # Use REDISTRIBUTED coefficients for curve plotting (after overlap absorption)
    # This ensures curves match the legend values (physically interpreted)
    coeffs_plot = results.get('_coeffs_plot', [0.0, 0.0, 0.0])
    am_var_plot = coeffs_plot[0]
    pm_var_plot = coeffs_plot[1]
    base_noise_var_plot = coeffs_plot[2] if len(coeffs_plot) > 2 else 0.0

    # Convert to degrees
    phase_bins_deg = phase_bin_centers_rad * 180 / np.pi
    phase_deg = np.mod(phase * 180 / np.pi, 360)

    # --- Axes Management ---
    if axes is not None:
        ax1, ax2 = axes if isinstance(axes, (tuple, list)) else axes.flatten()
    else:
        if ax is None:
            ax = plt.gca()
        fig = ax.get_figure()
        if hasattr(ax, 'get_subplotspec') and ax.get_subplotspec():
            gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=ax.get_subplotspec(), hspace=0.35)
            ax.remove()
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
        else:
            pos = ax.get_position()
            ax.remove()
            ax1 = fig.add_axes([pos.x0, pos.y0 + pos.height/2, pos.width, pos.height/2])
            ax2 = fig.add_axes([pos.x0, pos.y0, pos.width, pos.height/2])

    # --- Top Panel: Signal and Error vs Phase ---
    ax1_left = ax1
    ax1_right = ax1.twinx()

    lines_ax1 = []
    labels_ax1 = []

    if len(phase_deg) > 0 and len(fitted_signal) > 0:
        sort_idx = np.argsort(phase_deg)
        line_data, = ax1_left.plot(phase_deg[sort_idx], fitted_signal[sort_idx], 'k-', linewidth=2)
        lines_ax1.append(line_data)
        labels_ax1.append('Data')
        ax1_left.set_xlim([0, 360])
        s_min, s_max = np.min(fitted_signal), np.max(fitted_signal)
        margin = (s_max - s_min) * 0.10
        ax1_left.set_ylim([s_min - margin, s_max + margin])
        ax1_left.set_ylabel('Data', color='k')
        ax1_left.tick_params(axis='y', labelcolor='k')

    if len(phase_deg) > 0 and len(error) > 0:
        line_err, = ax1_right.plot(phase_deg, error, 'r.', markersize=2, alpha=0.5)
        lines_ax1.append(line_err)
        labels_ax1.append('Error')
        if len(bin_error_mean_v) > 0:
            line_mean, = ax1_right.plot(phase_bins_deg, bin_error_mean_v, 'b-', linewidth=2)
            lines_ax1.append(line_mean)
            labels_ax1.append('Error Mean')
        ax1_right.set_xlim([0, 360])

        # Smart Y-limits with 10% margin
        y_min, y_max = np.min(error), np.max(error)
        y_range = y_max - y_min
        margin = y_range * 0.1 if y_range != 0 else 1.0
        ax1_right.set_ylim([y_min - margin, y_max + margin])

        ax1_right.set_ylabel('Error', color='r')
        ax1_right.tick_params(axis='y', labelcolor='r')

    ax1.set_xlabel('Phase (deg)')
    if title:
        ax1.set_title(f'{title}\nSignal and Error vs Phase')
    else:
        ax1.set_title('Signal and Error vs Phase')
    ax1.grid(True, alpha=0.3)
    if lines_ax1:
        ax1.legend(lines_ax1, labels_ax1, loc='upper left', fontsize=8)

    # --- Bottom Panel: RMS Error Bar Chart with Fitted Curves ---
    if len(bin_error_rms_v) > 0 and len(phase_bin_centers_rad) > 0:
        bin_width = 360 / len(phase_bin_centers_rad)

        # Bar plot
        ax2.bar(phase_bins_deg, bin_error_rms_v, width=bin_width*0.8,
                color='skyblue', alpha=0.8, edgecolor='darkblue', linewidth=0.5)

        # Fitted curves: use REDISTRIBUTED coefficients (after overlap absorption)
        # Cosine basis: AM sensitivity = cos², PM sensitivity = sin²
        am_sen = np.cos(phase_bin_centers_rad) ** 2
        pm_sen = np.sin(phase_bin_centers_rad) ** 2

        # Curves use redistributed coefficients (physically interpreted values)
        am_curve = np.sqrt(am_var_plot * am_sen + base_noise_var_plot)
        pm_curve = np.sqrt(pm_var_plot * pm_sen + base_noise_var_plot)
        total_curve = np.sqrt(am_var_plot * am_sen + pm_var_plot * pm_sen + base_noise_var_plot)

        # Legend labels show REDISTRIBUTED physical values (after overlap adjustment)
        am_str = _format_value_with_unit(am_noise_rms_v)
        pm_str = _format_value_with_unit(pm_noise_rms_v)
        pm_rad_str = '0.00 urad' if pm_noise_rms_rad < 1e-12 else f'{pm_noise_rms_rad * 1e6:.2f} urad'
        base_noise_str = _format_value_with_unit(base_noise_rms_v)
        total_rms = results.get('total_rms_v', 0.0)
        total_str = _format_value_with_unit(total_rms)

        ax2.plot(phase_bins_deg, am_curve, 'b-', linewidth=2, label=f'AM = {am_str}')
        ax2.plot(phase_bins_deg, pm_curve, 'r-', linewidth=2, label=f'PM = {pm_str} ({pm_rad_str})')
        if include_base_noise:
            base_noise_curve = np.full_like(phase_bins_deg, np.sqrt(base_noise_var_plot))
            ax2.plot(phase_bins_deg, base_noise_curve, 'g-', linewidth=1.5, label=f'Base Noise = {base_noise_str}')
        ax2.plot(phase_bins_deg, total_curve, 'k--', linewidth=2, label=f'Total = {total_str}')

        ax2.set_xlim([0, 360])
        max_rms = np.nanmax(bin_error_rms_v)
        ax2.set_ylim([0, max_rms * 1.5])
        ax2.set_ylabel('RMS Error')
        ax2.set_title(f'RMS Error vs Phase\nModel Confidence: R²={r_squared_binned:.3f}')
        ax2.legend(loc='upper left', fontsize=9)

    ax2.set_xlabel('Phase (deg)')
    ax2.grid(True, alpha=0.3)

    if ax is None:
        plt.tight_layout()
