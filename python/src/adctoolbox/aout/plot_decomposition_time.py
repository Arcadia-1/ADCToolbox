"""Plot time domain decomposition for ADC harmonic analysis.

This module provides a pure visualization utility for plotting decomposed
harmonic components in the time domain, strictly adhering to the Single
Responsibility Principle. No calculations are performed - only plotting.
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpecFromSubplotSpec

def plot_decomposition_time(
    results: dict,
    signal: np.ndarray,
    n_cycles: float = 1.5,
    axes=None,
    ax=None,
    title: str = None
):
    """Create a time-domain plot of harmonic decomposition results.

    This is a pure visualization function that displays the signal and its
    decomposed components (fundamental, harmonics, and other errors).

    Parameters
    ----------
    results : dict
        Dictionary from decompose_harmonic_error() containing:
        - 'fundamental_signal': Reconstructed fundamental component
        - 'harmonic_signal': Harmonic distortion components (2nd through nth)
        - 'noise_residual': Other errors not captured by harmonics
        - 'fundamental_freq': Normalized fundamental frequency
    signal : np.ndarray
        Original signal data
    n_cycles : float, default=1.5
        Number of cycles to display in the time-domain plot
    axes : tuple or array, optional
        Tuple of (ax1, ax2) for top and bottom panels.
    ax : matplotlib.axes.Axes, optional
        Single axis to split into 2 panels.
        If None, uses plt.gca() and splits it.
    title : str, optional
        Custom title for the plot

    Notes
    -----
    - Two-panel layout: top panel shows signal and fitted sinewave, bottom panel shows decomposed errors
    - Top panel: signal (black 'x' markers) and fundamental sinewave (gray line)
    - Bottom panel: harmonic distortions (red line) and other errors (blue line)
    - Display range automatically limits to first few periods, centered on largest error
    - 10% margin applied to y-axis limits for better visualization
    """

    # Validate inputs
    required_keys = ['fundamental_signal', 'harmonic_signal', 'noise_residual', 'fundamental_freq']
    for key in required_keys:
        if key not in results:
            raise ValueError(f"results must contain '{key}' key")

    # Extract data from results dictionary
    fundamental_signal = results['fundamental_signal']
    harmonic_signal = results['harmonic_signal']
    noise_residual = results['noise_residual']
    fundamental_freq = results['fundamental_freq']

    # Set default parameters
    max_periods = n_cycles
    min_samples = 50

    # --- Axes Management ---
    if axes is not None:
        # Support both tuple (ax1, ax2) and numpy array [ax1, ax2]
        ax1, ax2 = axes if isinstance(axes, (tuple, list)) else axes.flatten()
    else:
        # Single axis (or None), get current axis and split it
        if ax is None:
            ax = plt.gca()

        # Split single axis into 2 rows
        fig = ax.get_figure()
        if hasattr(ax, 'get_subplotspec') and ax.get_subplotspec():
            gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=ax.get_subplotspec(), hspace=0.35)
            ax.remove()  # Remove original placeholder axis
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
        else:
            # Fallback for manual positioning
            pos = ax.get_position()
            ax.remove()
            ax1 = fig.add_axes([pos.x0, pos.y0 + pos.height/2, pos.width, pos.height/2])
            ax2 = fig.add_axes([pos.x0, pos.y0, pos.width, pos.height/2])

    # Compute total error for determining center point
    total_error = signal - fundamental_signal

    # Find the index with largest error amplitude
    max_err_idx = np.argmax(np.abs(total_error))

    # Calculate number of samples needed for the desired cycles
    n_samples = len(signal)
    if fundamental_freq > 0:
        samples_per_cycle = int(1.0 / fundamental_freq)
        n_samples_display = min(int(max_periods * samples_per_cycle), n_samples)
    else:
        n_samples_display = min(min_samples, n_samples)

    # Position the largest error at 1/3 from the left (not centered)
    one_third_display = n_samples_display // 3
    xlim_start = max(0, max_err_idx - one_third_display)
    xlim_end = min(n_samples, xlim_start + n_samples_display)

    # Adjust start if we hit the end
    if xlim_end == n_samples and xlim_end - xlim_start < n_samples_display:
        xlim_start = max(0, xlim_end - n_samples_display)

    # Prepare display data
    x_indices = np.arange(xlim_start + 1, xlim_end + 1)
    signal_display = signal[xlim_start:xlim_end]
    fundamental_display = fundamental_signal[xlim_start:xlim_end]
    harmonic_display = harmonic_signal[xlim_start:xlim_end]
    residual_display = noise_residual[xlim_start:xlim_end]

    # ======================================================================
    # Top Panel: Signal (data points) and Fitted Sinewave
    # ======================================================================
    ax1.plot(x_indices, signal_display, 'bx', label='signal', markersize=4, alpha=0.6)
    ax1.plot(x_indices, fundamental_display, '-', color=[0.5, 0.5, 0.5], label='fitted sinewave', linewidth=1.5)

    # Set X-axis limits (1-indexed for display)
    ax1.set_xlim([xlim_start + 1, xlim_end])

    # Set Y-axis limits with 10% margin
    signal_min, signal_max = np.min(signal_display), np.max(signal_display)
    signal_range = signal_max - signal_min
    margin = signal_range * 0.1 if signal_range != 0 else 0.1
    ax1.set_ylim([signal_min - margin, signal_max + margin])

    ax1.set_xlabel('Samples')
    ax1.set_ylabel('Signal')
    if title:
        ax1.set_title(f'{title}\nSignal and Fitted Sinewave')
    else:
        ax1.set_title('Signal and Fitted Sinewave')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ======================================================================
    # Bottom Panel: Decomposed Error (Harmonics + Other Errors)
    # ======================================================================
    # Plot other errors first (bottom layer) with thin line
    ax2.plot(x_indices, residual_display, 'b-', label='other errors', linewidth=0.5, alpha=0.8)
    # Plot harmonics on top (overlay) with thicker line
    ax2.plot(x_indices, harmonic_display, 'r-', label='harmonics', linewidth=1.5)

    # Set X-axis limits (1-indexed for display)
    ax2.set_xlim([xlim_start + 1, xlim_end])

    # Set Y-axis limits with 10% margin
    error_data = np.concatenate([harmonic_display, residual_display])
    error_min, error_max = np.min(error_data), np.max(error_data)
    error_range = error_max - error_min
    margin_err = error_range * 0.1 if error_range != 0 else 0.1
    ax2.set_ylim([error_min - margin_err, error_max + margin_err])

    ax2.set_xlabel('Samples')
    ax2.set_ylabel('Error')
    ax2.set_title('Decomposed Error')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Tight layout if we created the figure structure ourselves
    if ax is None:
        plt.tight_layout()
