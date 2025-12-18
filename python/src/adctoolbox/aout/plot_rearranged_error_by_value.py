"""Visualization for value-binned error analysis."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec

def plot_rearranged_error_by_value(results: dict, axes=None, ax=None, title: str = None):
    """
    Plot Mean Error (INL) and RMS Error (Noise) vs Bin Index.

    Creates a comprehensive visualization showing:
    - Top panel: Scatter of raw error vs bin index with mean error overlay
    - Bottom panel: RMS error vs bin index as bar chart

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
    n_bins = results['n_bins']

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
        ax1.scatter(bin_indices, error, alpha=0.2, s=1, color='red', 
                   rasterized=True, label='Raw Error')

        # Mean error line overlay (INL Profile)
        ax1.plot(bin_centers, error_mean, 'b-', linewidth=1.5, label='Mean Error (INL)')

        # Set axis limits
        ax1.set_xlim([-0.5, n_bins - 0.5])
        
        # Smart Y-limits
        y_min, y_max = np.min(error), np.max(error)
        y_range = y_max - y_min
        margin = y_range * 0.1 if y_range != 0 else 1.0
        ax1.set_ylim([y_min - margin, y_max + margin])

        ax1.set_ylabel('Error')
        if title:
            ax1.set_title(f'{title}\nSignal and Error vs Value')
        else:
            ax1.set_title('Signal and Error vs Value')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=8)
        # Hide x-labels for top plot to avoid clutter
        ax1.set_xticklabels([])

    # ======================================================================
    # Bottom Panel: RMS Error Bar Chart (Noise Profile)
    # ======================================================================
    if len(bin_centers) > 0:
        # Bar chart
        ax2.bar(bin_centers, error_rms, width=0.9,
                color='skyblue', alpha=0.8, edgecolor='darkblue', linewidth=0.3)

        # Set axis limits
        ax2.set_xlim([-0.5, n_bins - 0.5])
        ax2.set_ylim([0, np.nanmax(error_rms) * 1.15])

        ax2.set_xlabel('Bin Index')
        ax2.set_ylabel('RMS Error')
        ax2.set_title('RMS Error vs Value')
        ax2.grid(True, alpha=0.3, axis='y')

    # Tight layout if we created the figure structure ourselves
    if ax is None:
        plt.tight_layout()