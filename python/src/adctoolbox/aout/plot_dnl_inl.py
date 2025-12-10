"""
Pure DNL/INL plotting functionality.

This module provides plotting functions for DNL and INL that can be used
with pre-computed values from compute_inl_from_sine.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec


def plot_dnl_inl(code, dnl, inl, num_bits=None, show_label=True, show_title=True, ax=None, color_dnl='b', color_inl='b'):
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
    show_label : bool, default=True
        Show axis labels and grid
    show_title : bool, default=True
        Show subplot titles with DNL/INL min/max ranges
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, uses current axis (plt.gca())
    color_dnl : str, default='r'
        Color for DNL plot
    color_inl : str, default='b'
        Color for INL plot

    Returns
    -------
    axes : tuple of matplotlib.axes.Axes
        The axes objects [dnl_ax, inl_ax]
    """
    # Get the axis to split
    if ax is None:
        ax = plt.gca()
    fig = ax.get_figure()

    # If the selected axis is part of a grid, split it into 2 rows
    if hasattr(ax, 'get_subplotspec') and ax.get_subplotspec() is not None:
        # Get the subplot spec of the current axis
        subplotspec = ax.get_subplotspec()

        # Create a nested gridspec with 2 rows and spacing
        nested_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=subplotspec, hspace=0.4)

        # Remove the old axis
        ax.remove()

        # Create two new axes in place
        ax_dnl = fig.add_subplot(nested_gs[0])
        ax_inl = fig.add_subplot(nested_gs[1])
        axes = (ax_dnl, ax_inl)
    else:
        # Not in a grid, create a simple 2-row subplot
        fig, axes = plt.subplots(2, 1)

    # Plot DNL (top)
    _plot_single_curve(axes[0], code, dnl, num_bits, show_label, 'DNL (LSB)', color_dnl)

    # Plot INL (bottom)
    _plot_single_curve(axes[1], code, inl, num_bits, show_label, 'INL (LSB)', color_inl)

    # Set titles on subplots
    if show_title:
        dnl_min, dnl_max = np.min(dnl), np.max(dnl)
        inl_min, inl_max = np.min(inl), np.max(inl)

        axes[0].set_title(f'DNL = [{dnl_min:.2f}, {dnl_max:.2f}] LSB', fontweight='bold', fontsize=10)
        axes[1].set_title(f'INL = [{inl_min:.2f}, {inl_max:.2f}] LSB', fontweight='bold', fontsize=10)

    return axes


def _plot_single_curve(ax, code, data, num_bits=None, show_label=True, ylabel='Data (LSB)', color='r'):
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
    show_label : bool, default=True
        Show axis labels and grid
    ylabel : str, default='Data (LSB)'
        Y-axis label
    color : str, default='r'
        Line color
    """
    # Plot curve
    ax.plot(code, data, f'{color}-', linewidth=0.5)

    # Add reference lines
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axhline(-0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    if show_label:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Code (LSB)')
        ax.set_ylabel(ylabel)

    # Set x-axis limits
    if num_bits is not None:
        ax.set_xlim([0, 2**num_bits])
    else:
        ax.set_xlim([np.min(code), np.max(code)])

    # Set y-axis limits: minimum ±1, or 1.2x data range if larger
    data_min, data_max = np.min(data), np.max(data)
    data_range = max(abs(data_min), abs(data_max))
    if data_range <= 1.0:
        ax.set_ylim([-1, 1])
    else:
        ax.set_ylim([data_min * 1.2, data_max * 1.2])
