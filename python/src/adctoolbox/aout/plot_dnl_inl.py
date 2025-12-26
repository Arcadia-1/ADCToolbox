"""
Pure DNL/INL plotting functionality.

This module provides plotting functions for DNL and INL that can be used
with pre-computed values from compute_inl_from_sine.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec

def plot_dnl_inl(code, dnl, inl, num_bits=None, show_title=True, col_title=None, axes=None, ax=None, color_dnl='b', color_inl='b'):
    """
    Plot DNL and INL curves in a 2-row subplot layout.

    Parameters
    ----------
    code : array_like
        Code values (x-axis)
    dnl : array_like
        DNL values in LSB (y-axis)
    inl : array_like
        INL values in LSB (y-axis)
    num_bits : int, optional
        Number of bits for x-axis limits. If None, uses code range.
    show_title : bool, default=True
        Show subplot titles with DNL/INL min/max ranges
    col_title : str, optional
        Column title to display above DNL plot (e.g., "N = 2^10")
    axes : tuple of matplotlib.axes.Axes, optional
        Pre-made axes tuple (dnl_ax, inl_ax). If provided, uses these directly.
    ax : matplotlib.axes.Axes, optional
        Single axis to split into 2 rows (auto-nested GridSpec). If None and axes=None, uses current axis.
    color_dnl : str, default='r'
        Color for DNL plot
    color_inl : str, default='b'
        Color for INL plot

    Returns
    -------
    axes : tuple of matplotlib.axes.Axes
        The axes objects [dnl_ax, inl_ax]
    """
    # Use provided axes tuple, or split a single axis
    if axes is not None:
        # Use provided axes tuple directly
        ax_dnl, ax_inl = axes
    else:
        # Single axis (or None), get current axis and split it
        if ax is None:
            ax = plt.gca()
        fig = ax.get_figure()

        # Split the axis into 2 rows using GridSpec
        if hasattr(ax, 'get_subplotspec') and ax.get_subplotspec() is not None:
            # Axis is part of a grid, use nested GridSpec
            subplotspec = ax.get_subplotspec()
            nested_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=subplotspec, hspace=0.4)
            ax.remove()
            ax_dnl = fig.add_subplot(nested_gs[0])
            ax_inl = fig.add_subplot(nested_gs[1])
            axes = (ax_dnl, ax_inl)
        else:
            # Axis not in a grid, replace it with 2-row layout
            pos = ax.get_position()
            ax.remove()
            ax_dnl = fig.add_axes([pos.x0, pos.y0 + pos.height/2, pos.width, pos.height/2])
            ax_inl = fig.add_axes([pos.x0, pos.y0, pos.width, pos.height/2])
            axes = (ax_dnl, ax_inl)

    # Plot DNL (top)
    _plot_single_curve(axes[0], code, dnl, num_bits, 'DNL (LSB)', color_dnl)

    # Plot INL (bottom)
    _plot_single_curve(axes[1], code, inl, num_bits, 'INL (LSB)', color_inl)

    # Set titles on subplots
    if show_title:
        dnl_min, dnl_max = np.min(dnl), np.max(dnl)
        inl_min, inl_max = np.min(inl), np.max(inl)

        # Column title on DNL axis if provided
        if col_title is not None:
            axes[0].set_title(f'{col_title}\nDNL = [{dnl_min:.2f}, {dnl_max:.2f}] LSB', fontweight='bold', fontsize=10)
        else:
            axes[0].set_title(f'DNL = [{dnl_min:.2f}, {dnl_max:.2f}] LSB', fontweight='bold', fontsize=10)

        axes[1].set_title(f'INL = [{inl_min:.2f}, {inl_max:.2f}] LSB', fontweight='bold', fontsize=10)

    return axes

def _plot_single_curve(ax, code, data, num_bits=None, ylabel='Data (LSB)', color='r'):
    """
    Helper function to plot a single DNL or INL curve.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on
    code : array_like
        Code values (x-axis)
    data : array_like
        Data values in LSB (y-axis)
    num_bits : int, optional
        Number of bits for x-axis limits. If None, uses code range.
    ylabel : str, default='Data (LSB)'
        Y-axis label
    color : str, default='r'
        Line color
    """
    # Plot curve
    ax.plot(code, data, f'{color}-', linewidth=0.5)

    # Add reference lines
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1.2, alpha=0.8)
    ax.axhline(-1.0, color='gray', linestyle='--', linewidth=1.2, alpha=0.8)

    # Always show grid and labels
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Code (LSB)')
    ax.set_ylabel(ylabel)

    # Set x-axis limits
    if num_bits is not None:
        full_scale = 2**num_bits
        ax.set_xlim([0, full_scale])
        # Set x-ticks: 1/8, 2/8, ..., 8/8 of full scale with actual code values as labels
        tick_positions = [full_scale * i / 8 for i in range(1, 9)]
        tick_labels = [f'{int(full_scale * i / 8)}' for i in range(1, 9)]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
    else:
        ax.set_xlim([np.min(code), np.max(code)])

    # Set y-axis limits: minimum ±1, or 1.2x data range if larger
    data_min, data_max = np.min(data), np.max(data)
    data_range = max(abs(data_min), abs(data_max))
    if data_range <= 1.0:
        ax.set_ylim([-1, 1])
    else:
        ax.set_ylim([data_min * 1.2, data_max * 1.2])
