"""Plot two-tone spectrum - pure visualization module.

This module provides plotting functions for two-tone IMD analysis,
following the modular architecture pattern with separation of concerns.

Matches MATLAB specPlot2Tone.m annotation style.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict


def plot_two_tone_spectrum(
    analysis_results: Dict,
    harmonic: int = 7,
    ax: Optional[plt.Axes] = None,
    show_title: bool = True,
    show_labels: bool = True
) -> plt.Axes:
    """
    Plot two-tone spectrum with IMD products marked.

    Pure visualization function - no calculations performed.
    Matches MATLAB specPlot2Tone.m annotation style.

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
    spectrum_power = plot_data['spectrum_power']
    bin1 = plot_data['bin1']
    bin2 = plot_data['bin2']
    N = plot_data['N']
    fs = plot_data['fs']
    harmonic_products = plot_data['harmonic_products']

    # Setup axes
    if ax is None:
        ax = plt.gca()

    # Plot spectrum
    ax.plot(freq, spec_db, 'b-', linewidth=0.5, alpha=0.7)

    # Calculate min for text positioning (MATLAB: mins = min(10*log10(spec)))
    mins = np.min(spec_db[spec_db > -200])

    # Mark fundamental tone bins with red (MATLAB style)
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
                b = product['bin']
                order = product['order']
                # Text label with order number (MATLAB: fontsize=12)
                ax.text(freq[b], spec_db[b] + 5, str(order),
                       fontname='Arial', fontsize=12, ha='center', color='red')
                # Red line on bins (MATLAB: plot red line on b-2:b+2)
                b_start = max(b - 2, 0)
                b_end = min(b + 3, len(freq))
                ax.plot(freq[b_start:b_end], spec_db[b_start:b_end], 'r-', linewidth=1.5)

    # Add Nyquist line (MATLAB: plot([1,1]*Fs/2,[0,-mins],'--'))
    if show_labels:
        ax.axvline(fs / 2, color='k', linestyle='--', linewidth=1, alpha=0.5)

    # Add frequency and power labels for F1 and F2 (MATLAB: lines 114-122)
    if show_labels:
        pwr1 = metrics['signal_power_1_dbfs']
        pwr2 = metrics['signal_power_2_dbfs']
        freq1 = freq[bin1]
        freq2 = freq[bin2]

        # Format frequency display (K/M/G suffix like plot_spectrum.py)
        def format_freq(f):
            if f >= 1e9: return f'{f/1e9:.1f} GHz'
            elif f >= 1e6: return f'{f/1e6:.1f} MHz'
            elif f >= 1e3: return f'{f/1e3:.1f} kHz'
            else: return f'{f:.1f} Hz'

        freq1_str = format_freq(freq1)
        freq2_str = format_freq(freq2)

        # Position labels: left signal gets right-aligned label (on its left)
        #                  right signal gets left-aligned label (on its right)
        freq_span = fs / 2 - freq[1]
        x_offset = freq_span * 0.01  # 3% of frequency range

        # F1 is always < F2 (ensured in calculate function)
        # F1 label on left side of peak (right-aligned), positioned above peak
        ax.text(freq1 - x_offset, pwr1, freq1_str,
               ha='right', va='center', fontsize=10, color='red')
        ax.text(freq1 - x_offset, pwr1 - 5, f'{pwr1:.1f} dB',
               ha='right', va='center', fontsize=10, color='red')

        # F2 label on right side of peak (left-aligned), positioned above peak
        ax.text(freq2 + x_offset, pwr2, freq2_str,
               ha='left', va='center', fontsize=10, color='red')
        ax.text(freq2 + x_offset, pwr2 - 5, f'{pwr2:.1f} dB',
               ha='left', va='center', fontsize=10, color='red')

    # Add metrics text (MATLAB: lines 124-130)
    if show_labels:
        # Format frequency display for Fs (K/M/G suffix like plot_spectrum.py)
        def format_freq(f):
            if f >= 1e9: return f'{f/1e9:.1f}G'
            elif f >= 1e6: return f'{f/1e6:.1f}M'
            elif f >= 1e3: return f'{f/1e3:.1f}K'
            else: return f'{f:.1f}'

        fs_str = f'Fs = {format_freq(fs)} Hz'

        # Adaptive positioning: avoid signal peaks (like plot_spectrum.py)
        # Check if both tones are on the left side of spectrum
        if bin2 / N < 0.3:  # Both tones on left side
            x_pos = fs * 0.3  # Put metrics on right
        else:  # Tones on right or spread across
            x_pos = fs * 0.01  # Put metrics on left

        metrics_text = [
            fs_str,
            f"ENOB = {metrics['enob']:.2f}",
            f"SNDR = {metrics['sndr_db']:.2f} dB",
            f"SFDR = {metrics['sfdr_db']:.2f} dB",
            f"SNR = {metrics['snr_db']:.2f} dB",
            f"Noise Floor = {metrics['noise_floor_db']:.2f} dB",
            f"IMD2 = {metrics['imd2_db']:.2f} dB",
            f"IMD3 = {metrics['imd3_db']:.2f} dB"
        ]

        # Calculate y position based on plot range (like plot_spectrum.py)
        y_start = mins * 0.05
        y_step = mins * 0.05

        for i, text in enumerate(metrics_text):
            ax.text(x_pos, y_start + i * y_step, text, fontsize=10)

    # Configure axes (MATLAB: axis([Fs/N, Fs/2, mins, 0]))
    ax.set_xlabel('Freq (Hz)', fontsize=10)
    ax.set_ylabel('dBFS', fontsize=10)

    if show_title:
        ax.set_title('Power Spectrum', fontsize=12, fontweight='bold')

    ax.grid(True, alpha=0.3)
    ax.set_xlim([freq[1], fs / 2])

    # Set ylim based on noise floor (unified with plot_spectrum.py)
    # Adaptive y-axis: start at -100 dB, extend if >5% of data is below each threshold
    valid_spec_db = spec_db[spec_db > -200]
    minx = -100
    for threshold in [-100, -120, -140, -160, -180]:
        below_threshold = np.sum(valid_spec_db < threshold)
        percentage = below_threshold / len(valid_spec_db) * 100
        if percentage > 5.0:
            minx = threshold - 20  # Extend to next level
        else:
            break
    minx = max(minx, -200)  # Absolute floor
    ax.set_ylim([minx, 0])

    return ax
