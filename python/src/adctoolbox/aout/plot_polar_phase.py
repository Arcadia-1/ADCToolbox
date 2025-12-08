"""Plot polar phase spectrum for ADC analysis.

This module provides a pure visualization utility for creating polar plots
of complex spectra, strictly adhering to the Single Responsibility Principle.
No calculations or data processing are performed - only plotting.

Matches MATLAB plotphase.m behavior.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_polar_phase(plot_data, harmonic=5, ax=None):
    """Create a polar plot of complex spectrum data.

    This is a pure visualization function that transforms pre-computed complex
    spectrum data into a professional polar plot matching MATLAB plotphase.m.

    Parameters
    ----------
    plot_data : dict
        Dictionary containing all pre-computed elements required for plotting:

        - 'complex_spec_coherent': np.ndarray
            Complex spectrum array (phase-aligned) ready for plotting
        - 'minR_dB': float
            Noise floor level in dB for scaling the radial axis
        - 'bin_idx': int
            Index of the fundamental frequency bin
        - 'N_fft': int
            FFT length (for harmonic position calculations)

    harmonic : int, optional
        Number of harmonics to mark on the plot, by default 5

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
    - Radius scaling maps noise floor (minR_dB) to r=0 and 0 dBFS to max perimeter
    - Polar axes configured for ADC standards: theta zero at top, clockwise direction
    - Fundamental is normalized to 0 degrees (pointing up)
    - All spectrum points shown as black dots, harmonics marked in blue
    """

    # Validate inputs
    required_keys = ['complex_spec_coherent', 'minR_dB', 'bin_idx', 'N_fft']
    for key in required_keys:
        if key not in plot_data:
            raise ValueError(f"plot_data must contain '{key}' key")

    # Extract data from plot_data dictionary
    spec = plot_data['complex_spec_coherent']
    minR_dB = plot_data['minR_dB']
    bin_idx = plot_data['bin_idx']
    N_fft = plot_data['N_fft']

    # Create axes if not provided (must be polar projection)
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='polar')
    else:
        # Verify axes has polar projection
        if not hasattr(ax, 'set_theta_zero_location'):
            raise ValueError("Axes must have polar projection")

    # Get magnitude and phase
    phi = spec / (np.abs(spec) + 1e-20)  # Normalized phase
    mag_dB = 20 * np.log10(np.abs(spec) + 1e-20)

    # Normalize to noise floor
    mag_dB = np.maximum(mag_dB, minR_dB)
    radius = mag_dB - minR_dB

    # Create phase-weighted spectrum (matches MATLAB: spec = phi.*spec)
    spec_polar = phi * radius

    # Extract phase and magnitude
    phase = np.angle(spec_polar)
    mag = np.abs(spec_polar)

    # Plot all points as black dots (matches MATLAB: polarplot(spec,'k.'))
    ax.scatter(phase, mag, s=1, c='k', alpha=0.5, label='Spectrum')

    # Configure polar axes (matches MATLAB settings)
    ax.set_theta_zero_location('N')  # Theta zero at top
    ax.set_theta_direction(-1)  # Clockwise

    # Set radial axis limits and ticks (matches MATLAB)
    max_radius = -minR_dB
    tick_values = np.arange(0, max_radius + 1, 10)  # Every 10 dB
    ax.set_rticks(tick_values)
    ax.set_ylim([0, max_radius])

    # Set radial tick labels to show dB values (matches MATLAB RTickLabel)
    tick_labels = [str(int(minR_dB + val)) for val in tick_values]
    ax.set_yticklabels(tick_labels)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Mark fundamental tone (make it CLEARLY visible!)
    # Fundamental should be at phase=0 after alignment (pointing up)
    fundamental_bin = bin_idx
    if fundamental_bin < len(spec_polar):
        # Plot a FILLED circle at fundamental (more visible than hollow harmonics)
        ax.plot(phase[fundamental_bin], mag[fundamental_bin], 'o',
               markersize=12, markeredgecolor='blue', markerfacecolor='blue',
               markeredgewidth=2, label='1 (Fundamental)', zorder=10)
        # Draw THICK line from center to fundamental
        ax.plot([0, phase[fundamental_bin]], [0, mag[fundamental_bin]],
               'b-', linewidth=3, zorder=10)
        # Add "1" label for fundamental
        ax.text(phase[fundamental_bin] + 0.1, mag[fundamental_bin], '1',
               fontname='Arial', fontsize=10, ha='center', fontweight='bold')

    # Mark harmonics (matches MATLAB harmonic plotting)
    for h in range(2, harmonic + 1):
        # Calculate harmonic bin with aliasing (matches MATLAB alias function)
        harmonic_bin = (fundamental_bin * h) % N_fft

        # Handle negative frequency wrap-around for real signals
        if harmonic_bin > N_fft // 2:
            harmonic_bin = N_fft - harmonic_bin

        if harmonic_bin < len(spec_polar):
            # Plot harmonic marker (blue square, matches MATLAB 'bs')
            ax.plot(phase[harmonic_bin], mag[harmonic_bin], 's',
                   markersize=6, markeredgecolor='blue', markerfacecolor='none',
                   markeredgewidth=1.5)
            # Draw line from center
            ax.plot([0, phase[harmonic_bin]], [0, mag[harmonic_bin]],
                   'b-', linewidth=2)

            # Add harmonic number label (matches MATLAB text annotation)
            ax.text(phase[harmonic_bin] + 0.1, mag[harmonic_bin], str(h),
                   fontname='Arial', fontsize=8, ha='center')

    # Set title
    ax.set_title('Spectrum Phase (FFT)', pad=20, fontsize=12)

    return ax