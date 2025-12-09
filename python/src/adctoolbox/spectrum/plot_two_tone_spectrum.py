"""Plot two-tone spectrum - pure visualization module.

This module provides plotting functions for two-tone IMD analysis,
following the modular architecture pattern with separation of concerns.

Matches MATLAB specPlot2Tone.m annotation style.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict
from ..common import fold_bin_to_nyquist


def plot_two_tone_spectrum(
    analysis_results: Dict,
    harmonic: int = 7,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    show_metrics: bool = True,
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
    title : str, optional
        Custom title for the plot. If None, uses default title
    show_metrics : bool, optional
        Whether to display metrics text on plot (default: True)
    show_labels : bool, optional
        Whether to show frequency/power labels for tones (default: True)

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
    fs = plot_data['fs']

    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

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

    # Mark harmonics and IMD products (MATLAB style: lines 67-94)
    if harmonic > 0:
        for i in range(2, harmonic + 1):
            for jj in range(i + 1):
                # Positive combination: jj*f1 + (i-jj)*f2
                b = fold_bin_to_nyquist((bin1) * jj + (bin2) * (i - jj), N)
                if 0 < b < len(freq):
                    # Text label with order number (MATLAB: fontsize=12)
                    ax.text(freq[b], spec_db[b] + 5, str(i),
                           fontname='Arial', fontsize=12, ha='center', color='red')
                    # Red line on bins (MATLAB: plot red line on b-2:b+2)
                    b_start = max(b - 2, 0)
                    b_end = min(b + 3, len(freq))
                    ax.plot(freq[b_start:b_end], spec_db[b_start:b_end], 'r-', linewidth=1.5)

                # Negative combination 1: -jj*f1 + (i-jj)*f2
                if -(bin1) * jj + (bin2) * (i - jj) > 0:
                    b = fold_bin_to_nyquist(-(bin1) * jj + (bin2) * (i - jj), N)
                    if 0 < b < len(freq):
                        ax.text(freq[b], spec_db[b] + 5, str(i),
                               fontname='Arial', fontsize=12, ha='center', color='red')
                        b_start = max(b - 2, 0)
                        b_end = min(b + 3, len(freq))
                        ax.plot(freq[b_start:b_end], spec_db[b_start:b_end], 'r-', linewidth=1.5)

                # Negative combination 2: jj*f1 - (i-jj)*f2
                if (bin1) * jj - (bin2) * (i - jj) > 0:
                    b = fold_bin_to_nyquist((bin1) * jj - (bin2) * (i - jj), N)
                    if 0 < b < len(freq):
                        ax.text(freq[b], spec_db[b] + 5, str(i),
                               fontname='Arial', fontsize=12, ha='center', color='red')
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
        y_offset = -8

        # F1 is always < F2 (ensured in calculate function)
        # F1 label on left side of peak (right-aligned)
        ax.text(freq1, pwr1 + y_offset, freq1_str,
               ha='right', va='top', fontsize=10, color='red')
        ax.text(freq1, pwr1 + y_offset - 5, f'{pwr1:.1f} dB',
               ha='right', va='top', fontsize=10, color='red')

        # F2 label on right side of peak (left-aligned)
        ax.text(freq2, pwr2 + y_offset, freq2_str,
               ha='left', va='top', fontsize=10, color='red')
        ax.text(freq2, pwr2 + y_offset - 5, f'{pwr2:.1f} dB',
               ha='left', va='top', fontsize=10, color='red')

    # Add metrics text (MATLAB: lines 124-130)
    if show_metrics:
        # Format frequency display for Fs (K/M/G suffix like plot_spectrum.py)
        def format_freq(f):
            if f >= 1e9: return f'{f/1e9:.1f}G'
            elif f >= 1e6: return f'{f/1e6:.1f}M'
            elif f >= 1e3: return f'{f/1e3:.1f}K'
            else: return f'{f:.1f}'

        fs_str = f'Fs = {format_freq(fs)} Hz'

        # Position text on RIGHT side to avoid signal peaks on left
        x_pos = freq[-10]  # Near right edge
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

        # MATLAB spacing: mins*0.05, mins*0.10, ..., mins*0.40
        y_positions = [mins * (0.05 + 0.05 * i) for i in range(len(metrics_text))]

        for text, y_pos in zip(metrics_text, y_positions):
            ax.text(x_pos, y_pos, text, fontsize=10, ha='right')

    # Configure axes (MATLAB: axis([Fs/N, Fs/2, mins, 0]))
    ax.set_xlabel('Freq (Hz)', fontsize=10)
    ax.set_ylabel('dBFS', fontsize=10)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title('Output Spectrum', fontsize=14, fontweight='bold')

    ax.grid(True, alpha=0.3)
    ax.set_xlim([freq[1], fs / 2])

    # Set ylim based on noise floor (like plot_spectrum.py)
    # Round down to nearest -20 dB for clean axis
    minx = min(max(np.median(spec_db[spec_db > -200]) - 20, -200), -40)
    minx = np.floor(minx / 20) * 20  # Round to -120, -140, etc.
    ax.set_ylim([minx, 0])

    return ax
