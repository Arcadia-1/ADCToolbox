"""Plot time domain decomposition for ADC harmonic analysis.

This module provides a pure visualization utility for plotting decomposed
harmonic components in the time domain, strictly adhering to the Single
Responsibility Principle. No calculations are performed - only plotting.

Matches the visualization style from exp_a04_decompose_harmonics.py and
MATLAB tomdec.m plotting behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def plot_decomposition_time(plot_data: dict, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Create a time-domain plot of harmonic decomposition results (MATLAB tomdec.m style).

    This is a pure visualization function that displays the signal and its
    decomposed components (fundamental, harmonics, and other errors).

    Parameters
    ----------
    plot_data : dict
        Dictionary containing all pre-computed elements required for plotting:

        Required keys:
        - 'signal': np.ndarray
            Original signal data
        - 'fundamental_signal': np.ndarray
            Reconstructed fundamental component (matches 'sine' from MATLAB)
        - 'harmonic_signal': np.ndarray
            Harmonic distortion components, 2nd through nth (matches 'har' from MATLAB)
        - 'residual': np.ndarray
            Other errors not captured by harmonics (matches 'oth' from MATLAB)
        - 'fundamental_freq': float
            Normalized fundamental frequency (for determining display range)

        Optional keys:
        - 'title': str
            Custom title for the plot (default: 'Harmonic Decomposition')
        - 'max_periods': int
            Maximum number of periods to display (default: 1.5)
        - 'min_samples': int
            Minimum number of samples to display (default: 50)

    ax : matplotlib.axes.Axes, optional
        Pre-configured Matplotlib Axes object.
        If None, a new figure and axes will be created.

    Returns
    -------
    matplotlib.axes.Axes
        The configured axes object containing the plot

    Notes
    -----
    - Plotting style matches MATLAB tomdec.m: left y-axis for signal, right y-axis for errors
    - Left Y-axis (signal): signal (black 'x' markers) and fundamental sinewave (gray line)
    - Right Y-axis (error): harmonic distortions (red line) and other errors (blue line)
    - Display range automatically limits to first few periods using MATLAB algorithm
    - Legend shows all four components: signal, sinewave, harmonics, other errors
    """

    # Validate inputs
    required_keys = ['signal', 'fundamental_signal', 'harmonic_signal', 'residual', 'fundamental_freq']
    for key in required_keys:
        if key not in plot_data:
            raise ValueError(f"plot_data must contain '{key}' key")

    # Extract data from plot_data dictionary
    signal = plot_data['signal']
    sine = plot_data['fundamental_signal']  # Matches MATLAB naming
    har = plot_data['harmonic_signal']      # Matches MATLAB naming
    oth = plot_data['residual']              # Matches MATLAB naming
    fundamental_freq = plot_data['fundamental_freq']

    # Optional parameters (matching MATLAB defaults)
    title = plot_data.get('title', 'Harmonic Decomposition')
    max_periods = plot_data.get('max_periods', 1.5)
    min_samples = plot_data.get('min_samples', 50)

    # Create axes if not provided
    if ax is None:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)

    # Use yyaxis pattern from MATLAB (emulated with twinx)
    # LEFT Y-AXIS: Signal and fundamental sinewave
    ax.plot(signal, 'kx', label='signal', markersize=4, alpha=0.6)
    ax.plot(sine, '-', color=[0.5, 0.5, 0.5], label='sinewave', linewidth=1.5)

    # Determine display range using MATLAB algorithm:
    # xlim_max = min(max(1.5/freq, 50), length(sig))
    n_samples = len(signal)
    if fundamental_freq > 0:
        xlim_max = min(max(int(max_periods / fundamental_freq), min_samples), n_samples)
    else:
        xlim_max = min(min_samples, n_samples)

    ax.set_xlim([1, xlim_max])

    # Set Y-axis limits for signal (matching MATLAB: 1.1x expansion)
    signal_min, signal_max = np.min(signal), np.max(signal)
    ax.set_ylim([signal_min * 1.1, signal_max * 1.1])
    ax.set_ylabel('Signal', fontsize=11)
    ax.tick_params(axis='y', labelcolor='k')

    # Create right Y-axis for errors
    ax2 = ax.twinx()

    # RIGHT Y-AXIS: Harmonic and other error components
    ax2.plot(har, 'r-', label='harmonics', linewidth=1.5)
    ax2.plot(oth, 'b-', label='other errors', linewidth=1.5)

    # Compute total error for y-axis scaling
    err = signal - sine

    # Set Y-axis limits for errors (matching MATLAB: 1.1x expansion)
    error_min, error_max = np.min(err), np.max(err)
    ax2.set_ylim([error_min * 1.1, error_max * 1.1])
    ax2.set_ylabel('Error', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='r')

    # Labels and title
    ax.set_xlabel('Samples', fontsize=11)
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')

    # Merge legends from both axes (matching MATLAB legend order)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

    # Grid for readability
    ax.grid(True, alpha=0.3)

    return ax
