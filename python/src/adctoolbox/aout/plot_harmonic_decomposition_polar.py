"""
Plot harmonic decomposition results in polar (LMS) domain.

Visualization function for displaying harmonic components in polar coordinates
showing magnitude and phase relationships.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_harmonic_decomposition_polar(results, ax=None):
    """
    Plot harmonic decomposition results in polar domain (LMS mode).

    Parameters
    ----------
    results : dict
        Dictionary from compute_harmonic_decomposition()
    ax : matplotlib.axes.Axes, optional
        Polar axes to plot on. If None, creates new polar axes
    """
    data = results['data']
    normalized_freq = results['normalized_freq']
    n_samples = results['n_samples']
    order = results['order']

    # Create or use provided polar axes
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='polar')
    else:
        fig = ax.get_figure()

    # Compute I/Q components for fundamental and harmonics
    t = np.arange(n_samples)
    phase = t * normalized_freq * 2 * np.pi

    # Fundamental (1st harmonic)
    cos_1 = np.cos(phase)
    sin_1 = np.sin(phase)
    i_1 = np.mean(cos_1 * data) * 2
    q_1 = np.mean(sin_1 * data) * 2
    mag_1 = np.sqrt(i_1**2 + q_1**2)
    phase_1 = np.arctan2(q_1, i_1)

    # Plot fundamental
    ax.plot([phase_1, phase_1], [0, mag_1], 'b-', linewidth=2.5, label=f'Fundamental (k=1)')
    ax.plot(phase_1, mag_1, 'bo', markersize=8)

    # Compute and plot harmonics
    for k in range(2, order + 1):
        cos_k = np.cos(phase * k)
        sin_k = np.sin(phase * k)
        i_k = np.mean(cos_k * data) * 2
        q_k = np.mean(sin_k * data) * 2
        mag_k = np.sqrt(i_k**2 + q_k**2)
        phase_k = np.arctan2(q_k, i_k)

        # Plot harmonic with varying colors
        color = plt.cm.Spectral(k / (order + 1))
        ax.plot([phase_k, phase_k], [0, mag_k], color=color, linewidth=2, alpha=0.7)
        ax.plot(phase_k, mag_k, 'o', color=color, markersize=6)

    # Formatting
    ax.set_xlabel('Phase (rad)', fontsize=10, labelpad=20)
    ax.set_title(f'Harmonic Decomposition - Polar (LMS Mode)\nfreq={normalized_freq:.6f}, order={order}', fontsize=11, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)

    # Create legend with fundamental and harmonics
    lines = [plt.Line2D([0], [0], color='b', linewidth=2.5, marker='o', markersize=8, label='Fundamental')]
    for k in range(2, min(order + 1, 6)):  # Limit legend to avoid crowding
        color = plt.cm.Spectral(k / (order + 1))
        lines.append(plt.Line2D([0], [0], color=color, linewidth=2, marker='o', markersize=6, label=f'k={k}'))
    if order > 5:
        lines.append(plt.Line2D([0], [0], color='gray', linewidth=2, marker='o', markersize=6, label=f'k=6..{order}'))

    ax.legend(handles=lines, loc='upper right', fontsize=9)

    return ax
