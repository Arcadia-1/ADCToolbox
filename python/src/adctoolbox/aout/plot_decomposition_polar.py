"""Plot polar decomposition for ADC harmonic analysis (LMS mode).

This module provides a pure visualization utility for creating polar plots
of LMS-decomposed harmonics, strictly adhering to the Single Responsibility
Principle. No calculations are performed - only plotting.

Matches MATLAB plotphase.m LMS mode behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def plot_decomposition_polar(plot_data: dict, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Create a polar plot of LMS harmonic decomposition results.

    This is a pure visualization function that displays harmonics on a polar
    plot with a noise circle reference (matching MATLAB plotphase LMS mode).

    Parameters
    ----------
    plot_data : dict
        Dictionary containing all pre-computed elements required for plotting:

        Required keys:
        - 'harm_mag': np.ndarray
            Magnitude of each harmonic
        - 'harm_phase': np.ndarray
            Phase of each harmonic in radians (relative to fundamental)
        - 'harm_dB': np.ndarray
            Magnitude in dB relative to full scale
        - 'noise_dB': float
            Noise floor in dB (for noise circle)
        - 'harmonic': int
            Number of harmonics to display

        Optional keys:
        - 'title': str
            Custom title (default: 'Signal Component Phase (LMS)')
        - 'minR_dB': float
            Manual minimum radius in dB (auto-calculated if not provided)
        - 'maxR_dB': float
            Manual maximum radius in dB (auto-calculated if not provided)

    ax : matplotlib.axes.Axes, optional
        Pre-configured Matplotlib Axes object with polar projection.
        If None, a new figure and axes will be created.

    Returns
    -------
    matplotlib.axes.Axes
        The configured polar axes object containing the plot

    Notes
    -----
    - The function performs NO calculations, only visualization
    - Fundamental shown as filled blue circle (NOT red as originally in MATLAB)
    - Harmonics shown as hollow blue squares
    - Noise circle (dashed line) shows residual error level
    - Harmonics outside noise circle indicate significant distortion
    - Radius in dB with minR at center, maxR at perimeter
    - Polar axes: theta zero at top, clockwise direction
    """

    # Validate inputs
    required_keys = ['harm_mag', 'harm_phase', 'harm_dB', 'noise_dB', 'harmonic']
    for key in required_keys:
        if key not in plot_data:
            raise ValueError(f"plot_data must contain '{key}' key")

    # Extract data from plot_data dictionary
    harm_mag = plot_data['harm_mag']
    harm_phase = plot_data['harm_phase']
    harm_dB = plot_data['harm_dB']
    noise_dB = plot_data['noise_dB']
    harmonic = plot_data['harmonic']

    # Optional parameters
    title = plot_data.get('title', 'Signal Component Phase (LMS)')

    # Create axes if not provided (must be polar projection)
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='polar')
    else:
        # Verify axes has polar projection
        if not hasattr(ax, 'set_theta_zero_location'):
            raise ValueError("Axes must have polar projection")

    # Calculate axis limits (matches MATLAB logic)
    if 'maxR_dB' in plot_data and 'minR_dB' in plot_data:
        maxR_dB = plot_data['maxR_dB']
        minR_dB = plot_data['minR_dB']
    else:
        # Auto-calculate limits
        # Round maximum harmonic to nearest 10 dB
        maxR_dB = np.ceil(np.max(harm_dB) / 10) * 10
        # Set minimum to accommodate noise floor with margin
        minR_dB = min(np.min(harm_dB), noise_dB) - 10
        # Round minR to nearest 10 dB
        minR_dB = np.floor(minR_dB / 10) * 10

    # Convert to plot scale (radius = dB - minR_dB)
    harm_radius = harm_dB - minR_dB
    noise_radius = noise_dB - minR_dB

    # Configure polar axes (matches MATLAB settings)
    ax.set_theta_zero_location('N')  # Theta zero at top
    ax.set_theta_direction(-1)  # Clockwise

    # Draw noise circle (matches MATLAB: noise circle with label)
    theta_circle = np.linspace(0, 2 * np.pi, 100)
    ax.plot(theta_circle, noise_radius * np.ones_like(theta_circle),
            'k--', linewidth=1.5, label='Residual Noise')

    # Add noise circle label (matches MATLAB text annotation)
    ax.text(np.pi * 3 / 4, noise_radius,
            f'Residue Errors\n{noise_dB:.1f} dB',
            fontsize=9, color='k', ha='left')

    # Plot harmonics
    for ii in range(harmonic):
        if ii == 0:
            # Fundamental: filled blue circle (NOT red, as per user request)
            ax.plot(harm_phase[ii], harm_radius[ii], 'o',
                   markersize=12, markeredgecolor='blue', markerfacecolor='blue',
                   markeredgewidth=2, label='1 (Fundamental)', zorder=10)
            ax.plot([0, harm_phase[ii]], [0, harm_radius[ii]],
                   'b-', linewidth=3, zorder=10)
            # Add "1" label for fundamental
            ax.text(harm_phase[ii] + 0.1, harm_radius[ii], '1',
                   fontname='Arial', fontsize=10, ha='center', fontweight='bold')
        else:
            # Harmonics: hollow blue squares
            ax.plot(harm_phase[ii], harm_radius[ii], 's',
                   markersize=6, markeredgecolor='blue', markerfacecolor='none',
                   markeredgewidth=1.5)
            ax.plot([0, harm_phase[ii]], [0, harm_radius[ii]],
                   'b-', linewidth=2)
            # Add harmonic number label
            ax.text(harm_phase[ii] + 0.1, harm_radius[ii], str(ii + 1),
                   fontname='Arial', fontsize=10, ha='center')

    # Set radial axis limits and ticks (matches MATLAB)
    max_radius = maxR_dB - minR_dB
    tick_values = np.arange(0, max_radius + 1, 10)  # Every 10 dB
    ax.set_rticks(tick_values)
    ax.set_ylim([0, max_radius])

    # Set radial tick labels to show dB values (matches MATLAB RTickLabel)
    tick_labels = [str(int(minR_dB + val)) for val in tick_values]
    ax.set_yticklabels(tick_labels)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Set title
    ax.set_title(title, pad=20, fontsize=12, fontweight='bold')

    return ax
