"""Plot polar decomposition for ADC harmonic analysis.

This module provides a pure visualization utility for creating polar plots
of harmonic decomposition results, strictly adhering to the Single
Responsibility Principle. No calculations are performed - only plotting.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_decomposition_polar(
    results: dict,
    harmonic: int = 5,
    ax=None,
    title: str = None
) -> plt.Axes:
    """Create a polar plot of harmonic decomposition results.

    This is a pure visualization function that displays harmonics on a polar
    plot with a noise circle reference.

    Parameters
    ----------
    results : dict
        Dictionary from decompose_harmonic_error() containing:
        - 'magnitudes': Magnitude of each harmonic
        - 'phases': Phase of each harmonic in radians (relative to fundamental)
        - 'magnitudes_db': Magnitude in dB relative to full scale
        - 'noise_db': Noise floor in dB (for noise circle)
    harmonic : int, default=5
        Number of harmonics to display.
    ax : matplotlib.axes.Axes, optional
        Pre-configured Matplotlib Axes object with polar projection.
        If None, a new figure and axes will be created.
    title : str, optional
        Custom title for the plot.

    Returns
    -------
    matplotlib.axes.Axes
        The configured polar axes object containing the plot

    Notes
    -----
    - Fundamental shown as filled blue circle at phase 0
    - Harmonics shown as hollow blue squares
    - Noise circle (dashed line) shows residual error level
    - Harmonics outside noise circle indicate significant distortion
    - Radius in dB with automatic scaling
    - Polar axes: theta zero at top, clockwise direction
    """

    # Validate inputs
    required_keys = ['magnitudes', 'phases', 'magnitudes_db', 'noise_db']
    for key in required_keys:
        if key not in results:
            raise ValueError(f"results must contain '{key}' key")

    # Extract data from results dictionary
    magnitudes = results['magnitudes']
    phases = results['phases']
    magnitudes_db = results['magnitudes_db']
    noise_db = results['noise_db']

    # Limit to requested number of harmonics
    magnitudes = magnitudes[:harmonic]
    phases = phases[:harmonic]
    magnitudes_db = magnitudes_db[:harmonic]

    # --- Axes Management ---
    # Get current axis if not provided
    if ax is None:
        ax = plt.gca()

    # Ensure axis has polar projection
    if not hasattr(ax, 'set_theta_zero_location'):
        # Need to replace with polar axes
        fig = ax.get_figure()
        if hasattr(ax, 'get_subplotspec') and ax.get_subplotspec():
            # Use existing subplot position
            subplotspec = ax.get_subplotspec()
            ax.remove()
            ax = fig.add_subplot(subplotspec, projection='polar')
        else:
            # Fallback for manual positioning
            pos = ax.get_position()
            ax.remove()
            ax = fig.add_axes([pos.x0, pos.y0, pos.width, pos.height], projection='polar')

    # Calculate axis limits
    # Round maximum harmonic to nearest 10 dB
    maxR_dB = np.ceil(np.max(magnitudes_db) / 10) * 10
    # Set minimum to accommodate noise floor with margin
    minR_dB = min(np.min(magnitudes_db), noise_db) - 10
    # Round minR to nearest 10 dB
    minR_dB = np.floor(minR_dB / 10) * 10

    # Convert to plot scale (radius = dB - minR_dB)
    harm_radius = magnitudes_db - minR_dB
    noise_radius = noise_db - minR_dB

    # Configure polar axes
    ax.set_theta_zero_location('N')  # Theta zero at top
    ax.set_theta_direction(-1)  # Clockwise

    # Draw noise circle
    theta_circle = np.linspace(0, 2 * np.pi, 100)
    ax.plot(theta_circle, noise_radius * np.ones_like(theta_circle),
            'k--', linewidth=1.5, label='Residual Noise')

    # Add noise circle label at bottom
    ax.text(np.pi, noise_radius*1.1,
            f'Residue Errors\n{noise_db:.2f} dB',
            fontsize=11, color='k', ha='center', va='top')

    # Plot harmonics
    for i in range(harmonic):
        if i == 0:
            # Fundamental: filled blue circle
            ax.plot(phases[i], harm_radius[i], 'o',
                   markersize=12, markeredgecolor='blue', markerfacecolor='blue',
                   markeredgewidth=2, label='1 (Fundamental)', zorder=10)
            ax.plot([0, phases[i]], [0, harm_radius[i]],
                   'b-', linewidth=3, zorder=10)
            # Add "1" label for fundamental
            ax.text(phases[i] + 0.1, harm_radius[i], '1',
                   fontname='Arial', fontsize=10, ha='center', fontweight='bold')
        else:
            # Harmonics: hollow blue squares
            ax.plot(phases[i], harm_radius[i], 's',
                   markersize=6, markeredgecolor='blue', markerfacecolor='none',
                   markeredgewidth=1.5)
            ax.plot([0, phases[i]], [0, harm_radius[i]],
                   'b-', linewidth=2)
            # Add harmonic number label
            ax.text(phases[i] + 0.1, harm_radius[i], str(i + 1),
                   fontname='Arial', fontsize=10, ha='center')

    # Set radial axis limits and ticks
    max_radius = maxR_dB - minR_dB
    tick_values = np.arange(0, max_radius + 1, 10)  # Every 10 dB
    ax.set_rticks(tick_values)
    ax.set_ylim([0, max_radius])

    # Set radial tick labels to show dB values
    tick_labels = [str(int(minR_dB + val)) for val in tick_values]
    ax.set_yticklabels(tick_labels)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Set title
    if title:
        ax.set_title(f'{title}\nSignal Component Phase')
    else:
        ax.set_title('Signal Component Phase')

    return ax
