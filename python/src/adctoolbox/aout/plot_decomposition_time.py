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
    """Create a time-domain plot of harmonic decomposition results.

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
            Reconstructed fundamental component
        - 'harmonic_signal': np.ndarray
            Harmonic distortion components (2nd through nth)
        - 'residual': np.ndarray
            Other errors (noise, non-harmonic distortion)
        - 'fundamental_freq': float
            Normalized fundamental frequency (for determining display range)

        Optional keys:
        - 'title': str
            Custom title for the plot (default: 'Harmonic Decomposition')
        - 'max_periods': int
            Maximum number of periods to display (default: 3)
        - 'min_samples': int
            Minimum number of samples to display (default: 100)

    ax : matplotlib.axes.Axes, optional
        Pre-configured Matplotlib Axes object.
        If None, a new figure and axes will be created.

    Returns
    -------
    matplotlib.axes.Axes
        The configured axes object containing the plot

    Notes
    -----
    - The function performs NO calculations, only visualization
    - Left Y-axis shows signal values
    - Right Y-axis shows error values
    - Automatic unit selection (uV, mV, or V) based on error magnitude
    - Legend includes RMS values and power percentages
    - Display range limited to first few periods for clarity
    """

    # Validate inputs
    required_keys = ['signal', 'fundamental_signal', 'harmonic_signal', 'residual', 'fundamental_freq']
    for key in required_keys:
        if key not in plot_data:
            raise ValueError(f"plot_data must contain '{key}' key")

    # Extract data from plot_data dictionary
    signal = plot_data['signal']
    fundamental_signal = plot_data['fundamental_signal']
    harmonic_signal = plot_data['harmonic_signal']
    residual = plot_data['residual']
    fundamental_freq = plot_data['fundamental_freq']

    # Optional parameters
    title = plot_data.get('title', 'Harmonic Decomposition')
    max_periods = plot_data.get('max_periods', 3)
    min_samples = plot_data.get('min_samples', 100)

    # Create axes if not provided
    if ax is None:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)

    # Calculate total error
    total_error = signal - fundamental_signal

    # Calculate RMS values
    rms_harmonic = np.sqrt(np.mean(harmonic_signal**2))
    rms_residual = np.sqrt(np.mean(residual**2))
    rms_total = np.sqrt(np.mean(total_error**2))

    # Determine appropriate unit (uV, mV, or V)
    if rms_total < 1e-3:
        unit = 'uV'
        scale = 1e6
    elif rms_total < 1:
        unit = 'mV'
        scale = 1e3
    else:
        unit = 'V'
        scale = 1

    # Calculate power percentages (RMS^2 / Total^2)
    if rms_total > 0:
        harmonic_pct = (rms_harmonic / rms_total)**2 * 100
        residual_pct = (rms_residual / rms_total)**2 * 100
    else:
        harmonic_pct = 0
        residual_pct = 0

    # Plot signal on left Y-axis
    ax.plot(signal, 'kx', label='signal', markersize=3, alpha=0.5)
    ax.plot(fundamental_signal, '-', color=[0.5, 0.5, 0.5],
            label='fundamental', linewidth=1.5)

    # Determine display range (show first few periods or minimum samples)
    n_samples = len(signal)
    if fundamental_freq > 0:
        xlim_max = min(max(int(max_periods / fundamental_freq), min_samples), n_samples)
    else:
        xlim_max = min(min_samples, n_samples)

    ax.set_xlim([0, xlim_max])

    # Set Y-axis limits for signal
    signal_min, signal_max = np.min(signal), np.max(signal)
    ax.set_ylim([signal_min * 1.1, signal_max * 1.1])
    ax.set_ylabel('Signal', color='k', fontsize=11)
    ax.tick_params(axis='y', labelcolor='k')

    # Create right Y-axis for errors
    ax2 = ax.twinx()

    # Plot errors on right Y-axis
    ax2.plot(harmonic_signal, 'r-',
             label=f'harmonics ({rms_harmonic*scale:.1f}{unit}, {harmonic_pct:.1f}%)',
             linewidth=1.5)
    ax2.plot(residual, 'b-',
             label=f'other errors ({rms_residual*scale:.1f}{unit}, {residual_pct:.1f}%)',
             linewidth=1)

    # Set Y-axis limits for errors
    error_min, error_max = np.min(total_error), np.max(total_error)
    ax2.set_ylim([error_min * 1.1, error_max * 1.1])
    ax2.set_ylabel('Error', color='r', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='r')

    # Labels and title
    ax.set_xlabel('Samples', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')

    # Merge legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

    # Grid
    ax.grid(True, alpha=0.3)

    return ax
