"""Visualization for value-binned error analysis."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec

def plot_rearranged_error_by_value(results: dict, axes=None, ax=None, title: str = None):
    """
    Plot value-binned mean residual and RMS residual.

    Creates a comprehensive visualization showing:
    - Top panel: Scatter of raw residual vs value bin with mean residual overlay
    - Bottom panel: RMS residual vs value bin as bar chart

    Parameters
    ----------
    results : dict
        Dictionary from rearrange_error_by_value().
    axes : tuple or array, optional
        Tuple of (ax1, ax2) for top and bottom panels.
    ax : matplotlib.axes.Axes, optional
        Single axis to split into 2 panels.
    title : str, optional
        Test setup description for title.
    """
    
    # --- 1. Extract Data (Using new simplified keys) ---
    error = results.get('error', np.array([]))
    bin_indices = results.get('bin_indices', np.array([]))
    
    error_mean = results['error_mean']
    error_rms = results['error_rms']
    bin_centers = results['bin_centers']
    value_bin_centers = results.get('value_bin_centers', bin_centers)
    n_bins = results['n_bins']
    scatter_x = value_bin_centers[bin_indices]
    finite_value_bins = np.isfinite(value_bin_centers)

    if len(bin_indices) == 0: return

    # --- 2. Axes Management ---
    if axes is not None:
        # Support both tuple (ax1, ax2) and numpy array [ax1, ax2]
        # This makes it compatible with axes = plt.subplots(2, 1)[1]
        ax1, ax2 = axes if isinstance(axes, (tuple, list)) else axes.flatten()
    else:
        # Single axis (or None), get current axis and split it
        if ax is None:
            ax = plt.gca()
        
        # Split single axis into 2 rows
        fig = ax.get_figure()
        if hasattr(ax, 'get_subplotspec') and ax.get_subplotspec():
            gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=ax.get_subplotspec(), hspace=0.35)
            ax.remove() # Remove original placeholder axis
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
        else:
            # Fallback for manual positioning
            pos = ax.get_position()
            ax.remove()
            ax1 = fig.add_axes([pos.x0, pos.y0 + pos.height/2, pos.width, pos.height/2])
            ax2 = fig.add_axes([pos.x0, pos.y0, pos.width, pos.height/2])

    # ======================================================================
    # Top Panel: Scatter of error vs bin index with mean error overlay
    # ======================================================================
    if len(error) > 0:

        # Scatter plot: High transparency, rasterized for performance on large datasets
        ax1.scatter(scatter_x, error, alpha=0.2, s=1, color='red',
                   rasterized=True, label='Raw Residual')

        # Mean residual line overlay (value-binned conditional mean)
        ax1.plot(value_bin_centers, error_mean, 'b-', linewidth=1.5,
                 label='Value-Binned Mean Residual')

        # Set axis limits
        if np.any(finite_value_bins):
            x_min = np.nanmin(value_bin_centers)
            x_max = np.nanmax(value_bin_centers)
            x_margin = 0.02 * (x_max - x_min) if x_max > x_min else 0.5
            ax1.set_xlim([x_min - x_margin, x_max + x_margin])

        # Smart Y-limits
        y_min, y_max = np.min(error), np.max(error)
        y_range = y_max - y_min
        margin = y_range * 0.1 if y_range != 0 else 1.0
        ax1.set_ylim([y_min - margin, y_max + margin])

        ax1.set_ylabel('Residual')
        if title:
            ax1.set_title(title)
        else:
            ax1.set_title('Value-Binned Residual Diagnostic')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=8)
        # Hide x-labels for top plot to avoid clutter
        ax1.set_xticklabels([])

    # ======================================================================
    # Bottom Panel: RMS Error Bar Chart (Noise Profile)
    # ======================================================================
    if len(bin_centers) > 0:
        # Bar chart
        if len(value_bin_centers) > 1:
            bar_width = 0.9 * np.nanmedian(np.diff(value_bin_centers))
        else:
            bar_width = 0.9
        ax2.bar(value_bin_centers, error_rms, width=bar_width,
                color='skyblue', alpha=0.8, edgecolor='darkblue', linewidth=0.3)

        # Set axis limits
        if np.any(finite_value_bins):
            x_min = np.nanmin(value_bin_centers)
            x_max = np.nanmax(value_bin_centers)
            x_margin = 0.02 * (x_max - x_min) if x_max > x_min else 0.5
            ax2.set_xlim([x_min - x_margin, x_max + x_margin])
        ax2.set_ylim([0, np.nanmax(error_rms) * 1.15])

        ax2.set_xlabel('Signal Value')
        ax2.set_ylabel('RMS Residual')
        ax2.set_title('Value-Binned RMS Residual')
        ax2.grid(True, alpha=0.3, axis='y')

    # Tight layout if we created the figure structure ourselves
    if ax is None:
        plt.tight_layout()
