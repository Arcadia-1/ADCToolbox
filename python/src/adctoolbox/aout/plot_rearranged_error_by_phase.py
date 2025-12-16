"""
Plot phase error analysis results (RMS curves, AM/PM decomposition).

Visualization function for displaying phase-binned error analysis results,
including RMS curves and AM/PM noise separation.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_rearranged_error_by_phase(results: dict, ax=None):
    """
    Plot phase error analysis results (RMS curves, AM/PM).

    Creates a comprehensive visualization showing:
    - Top panel: Mean error vs phase
    - Bottom panel: RMS error vs phase with AM/PM fitted curves

    Parameters
    ----------
    results : dict
        Dictionary from rearrange_error_by_phase() with mode="binned" or mode="raw". Must contain:
        - 'erms': RMS error per phase bin
        - 'emean': Mean error per phase bin
        - 'phase_bins': Phase bin centers in radians
        - 'am_param': AM parameter
        - 'pm_param': PM parameter
        - 'baseline': Baseline noise
        - 'fundamental_amplitude': Fitted amplitude (optional)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure with 2 subplots.

    Notes
    -----
    The bottom panel shows the RMS error decomposition:
    - Blue curve: AM component (cos² dependence)
    - Red curve: PM component (sin² dependence)
    - Bars: Actual binned RMS values

    The fitted model is:
        RMS²(φ) = am_param² * cos²(φ) + pm_param² * A² * sin²(φ) + baseline

    Examples
    --------
    >>> from adctoolbox.aout import rearrange_error_by_phase
    >>> sig = np.sin(2*np.pi*0.1*np.arange(1000)) + 0.01*np.random.randn(1000)
    >>> results = rearrange_error_by_phase(sig, 0.1, mode="binned")
    >>> plot_error_binned_phase(results)
    """
    # Extract data from results
    erms = results['erms']
    emean = results['emean']
    phase_bins = results['phase_bins']
    am_param = results['am_param']
    pm_param = results['pm_param']
    baseline = results['baseline']
    fundamental_amplitude = results.get('fundamental_amplitude', 1.0)

    # Convert phase to degrees for plotting
    phase_bins_deg = phase_bins * 180 / np.pi

    # Create figure if no axes provided
    if ax is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    else:
        # Split provided axis into 2 subplots
        fig = ax.get_figure()
        pos = ax.get_position()
        ax.remove()
        ax1 = fig.add_axes([pos.x0, pos.y0 + pos.height/2, pos.width, pos.height/2])
        ax2 = fig.add_axes([pos.x0, pos.y0, pos.width, pos.height/2])

    # --- Top Panel: Mean Error vs Phase ---
    valid_mask = ~np.isnan(emean)
    ax1.plot(phase_bins_deg[valid_mask], emean[valid_mask], 'b-', linewidth=2, label='Mean Error')
    ax1.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlim([0, 360])
    ax1.set_ylabel('Mean Error')
    ax1.set_title('Phase Error Analysis: Mean Error vs Phase')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

    # --- Bottom Panel: RMS Error vs Phase with AM/PM curves ---
    valid_mask = ~np.isnan(erms)
    bin_width = 360 / len(phase_bins)

    # Bar plot for actual RMS
    ax2.bar(phase_bins_deg, erms, width=bin_width*0.8, color='skyblue',
            alpha=0.7, label='Binned RMS')

    # Compute fitted curves
    # AM curve: sqrt(am² * cos²(φ) + baseline)
    # PM curve: sqrt(pm² * A² * sin²(φ) + baseline)
    phase_dense = np.linspace(0, 2*np.pi, 360)
    phase_dense_deg = phase_dense * 180 / np.pi

    am_sensitivity = np.cos(phase_dense)**2
    pm_sensitivity = np.sin(phase_dense)**2

    am_curve_sq = am_param**2 * am_sensitivity + baseline
    pm_curve_sq = pm_param**2 * fundamental_amplitude**2 * pm_sensitivity + baseline

    # Only plot where values are non-negative
    am_curve = np.sqrt(np.maximum(am_curve_sq, 0))
    pm_curve = np.sqrt(np.maximum(pm_curve_sq, 0))

    ax2.plot(phase_dense_deg, am_curve, 'b-', linewidth=2, label='AM Component')
    ax2.plot(phase_dense_deg, pm_curve, 'r-', linewidth=2, label='PM Component')

    ax2.set_xlim([0, 360])
    ax2.set_ylim([0, np.nanmax(erms)*1.2 if np.any(valid_mask) else 1.0])
    ax2.set_xlabel('Phase (deg)')
    ax2.set_ylabel('RMS Error')

    # Add parameter annotations
    max_rms = np.nanmax(erms) if np.any(valid_mask) else 1.0
    text_y1 = max_rms * 1.15
    text_y2 = max_rms * 1.05

    # Normalize AM parameter by amplitude for display
    am_normalized = am_param / fundamental_amplitude if fundamental_amplitude > 1e-10 else am_param

    ax2.text(10, text_y1,
            f'Normalized AM Noise RMS = {am_normalized:.2e}',
            color='b', fontsize=10, fontweight='bold')
    ax2.text(10, text_y2,
            f'PM Noise RMS = {pm_param:.2e} rad',
            color='r', fontsize=10, fontweight='bold')

    ax2.set_title('RMS Error vs Phase (AM/PM Decomposition)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    plt.tight_layout()
