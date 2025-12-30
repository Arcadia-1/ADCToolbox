"""Plot two-tone spectrum - pure visualization module.

This module provides plotting functions for two-tone IMD analysis,
following the modular architecture pattern with separation of concerns.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_two_tone_spectrum(
    analysis_results: dict,
    harmonic: int = 7,
    ax=None,
    show_title: bool = True,
    show_labels: bool = True
) -> plt.Axes:
    """
    Plot two-tone spectrum with IMD products marked.

    Pure visualization function - no calculations performed.

    Parameters
    ----------
    analysis_results : dict
        Results from compute_two_tone_spectrum()
        Required keys:
        - 'plot_data': dict with 'freq', 'spec_db', 'bin1', 'bin2', 'N', 'fs'
        - 'metrics': dict with performance metrics
        - 'imd_bins': dict with IMD product bin locations (optional)
    harmonic : int, optional
        Number of harmonic orders to mark (default: 7)
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes to plot on. If None, creates new figure
    show_title : bool, optional
        Display title (default: True)
    show_labels : bool, optional
        Add labels and annotations (default: True)

    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the plot
    """
    # Extract data
    plot_data = analysis_results['plot_data']
    metrics = analysis_results['metrics']

    freq = plot_data['freq']
    spec_db = plot_data['spec_db']
    bin1 = plot_data['bin1']
    bin2 = plot_data['bin2']
    N = plot_data['N']
    M = plot_data.get('M', 1)  # Number of runs
    fs = plot_data['fs']
    harmonic_products = plot_data['harmonic_products']
    coherent_averaging = plot_data.get('coherent_averaging', False)

    # Setup axes
    if ax is None:
        ax = plt.gca()

    # Helper function for frequency formatting
    def format_freq(f: float) -> str:
        """Format frequency with appropriate SI prefix."""
        if f >= 1e9:
            return f'{f/1e9:.1f} GHz'
        elif f >= 1e6:
            return f'{f/1e6:.1f} MHz'
        elif f >= 1e3:
            return f'{f/1e3:.1f} kHz'
        else:
            return f'{f:.1f} Hz'

    def format_freq_short(f: float) -> str:
        """Format frequency with short SI prefix (for axis labels)."""
        if f >= 1e9:
            return f'{f/1e9:.1f}G'
        elif f >= 1e6:
            return f'{f/1e6:.1f}M'
        elif f >= 1e3:
            return f'{f/1e3:.1f}K'
        else:
            return f'{f:.1f}'

    # Plot spectrum
    ax.plot(freq, spec_db, 'b-', linewidth=0.5, alpha=0.7)

    # Calculate minimum for text positioning
    min_db = np.min(spec_db[spec_db > -200])

    # Mark fundamental tone bins
    if show_labels:
        # F1 bins
        f1_start = max(bin1 - 1, 0)
        f1_end = min(bin1 + 2, len(freq))
        ax.plot(freq[f1_start:f1_end], spec_db[f1_start:f1_end], 'r-', linewidth=1.5)

        # F2 bins
        f2_start = max(bin2 - 1, 0)
        f2_end = min(bin2 + 2, len(freq))
        ax.plot(freq[f2_start:f2_end], spec_db[f2_start:f2_end], 'r-', linewidth=1.5)

    # Mark harmonics and IMD products using pre-calculated data
    if harmonic > 0:
        for product in harmonic_products:
            if product['order'] <= harmonic:
                bin_idx = product['bin']
                order = product['order']
                # Text label with order number
                ax.text(freq[bin_idx], spec_db[bin_idx] + 5, str(order),
                       fontname='Arial', fontsize=12, ha='center', color='red')
                # Highlight bins around the product
                bin_start = max(bin_idx - 2, 0)
                bin_end = min(bin_idx + 3, len(freq))
                ax.plot(freq[bin_start:bin_end], spec_db[bin_start:bin_end], 'r-', linewidth=1.5)

    # Add Nyquist frequency line
    if show_labels:
        ax.axvline(fs / 2, color='k', linestyle='--', linewidth=1, alpha=0.5)

    # Add frequency and power labels for F1 and F2
    if show_labels:
        power_1_dbfs = metrics['signal_power_1_dbfs']
        power_2_dbfs = metrics['signal_power_2_dbfs']
        freq_1 = freq[bin1]
        freq_2 = freq[bin2]

        freq_1_str = format_freq(freq_1)
        freq_2_str = format_freq(freq_2)

        # Position labels: left signal gets right-aligned label, right signal gets left-aligned
        freq_span = fs / 2 - freq[1]
        x_offset = freq_span * 0.01  # 1% of frequency range

        # F1 is always < F2 (ensured in calculate function)
        # F1 label on left side of peak (right-aligned)
        ax.text(freq_1 - x_offset, power_1_dbfs, freq_1_str,
               ha='right', va='center', fontsize=10, color='red')
        ax.text(freq_1 - x_offset, power_1_dbfs - 5, f'{power_1_dbfs:.1f} dB',
               ha='right', va='center', fontsize=10, color='red')

        # F2 label on right side of peak (left-aligned)
        ax.text(freq_2 + x_offset, power_2_dbfs, freq_2_str,
               ha='left', va='center', fontsize=10, color='red')
        ax.text(freq_2 + x_offset, power_2_dbfs - 5, f'{power_2_dbfs:.1f} dB',
               ha='left', va='center', fontsize=10, color='red')

    # Add metrics text
    if show_labels:
        fs_str = f'Fs = {format_freq_short(fs)} Hz'

        # Adaptive positioning: avoid signal peaks
        # Check if both tones are on the left side of spectrum
        if bin2 / N < 0.3:
            x_pos = fs * 0.3  # Put metrics on right
        else:
            x_pos = fs * 0.01  # Put metrics on left

        metrics_text = [
            fs_str,
            f"ENOB = {metrics['enob']:.2f}",
            f"SNDR = {metrics['sndr_dbc']:.2f} dB",
            f"SFDR = {metrics['sfdr_dbc']:.2f} dB",
            f"SNR = {metrics['snr_dbc']:.2f} dB",
            f"Noise Floor = {metrics['noise_floor_db']:.2f} dB",
            f"IMD2 = {metrics['imd2_dbc']:.2f} dB",
            f"IMD3 = {metrics['imd3_dbc']:.2f} dB"
        ]

        # Calculate y position based on plot range
        y_start = min_db * 0.05
        y_step = min_db * 0.05

        for i, text in enumerate(metrics_text):
            ax.text(x_pos, y_start + i * y_step, text, fontsize=10)

    # Configure axes
    ax.set_xlabel('Freq (Hz)', fontsize=10)
    ax.set_ylabel('dBFS', fontsize=10)

    # Auto-generate title based on averaging mode
    if show_title:
        if M > 1:
            if coherent_averaging:
                ax.set_title(f'Coherent averaging (N_run = {M})', fontsize=12, fontweight='bold')
            else:
                ax.set_title(f'Power averaging (N_run = {M})', fontsize=12, fontweight='bold')
        else:
            ax.set_title('Power Spectrum', fontsize=12, fontweight='bold')

    ax.grid(True, alpha=0.3)
    ax.set_xlim([freq[1], fs / 2])

    # Adaptive y-axis limits based on noise floor
    # Start at -100 dB, extend if >5% of data is below each threshold
    valid_spec_db = spec_db[spec_db > -200]
    y_min = -100
    for threshold in [-100, -120, -140, -160, -180]:
        below_threshold = np.sum(valid_spec_db < threshold)
        percentage = below_threshold / len(valid_spec_db) * 100
        if percentage > 5.0:
            y_min = threshold - 20  # Extend to next level
        else:
            break
    y_min = max(y_min, -200)  # Absolute floor
    ax.set_ylim([y_min, 0])

    return ax
