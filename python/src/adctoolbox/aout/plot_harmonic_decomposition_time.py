"""
Plot harmonic decomposition results in time domain.

Visualization function for displaying computed harmonic decomposition results.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_harmonic_decomposition_time(results, ax=None):
    """
    Plot harmonic decomposition results in time domain.

    Parameters
    ----------
    results : dict
        Dictionary from compute_harmonic_decomposition()
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, uses plt.gca()
    """
    if ax is None:
        ax = plt.gca()

    data = results['data']
    fundamental_signal = results['fundamental_signal']
    harmonic_error = results['harmonic_error']
    other_error = results['other_error']
    total_error = results['total_error']
    normalized_freq = results['normalized_freq']
    n_samples = results['n_samples']
    order = results['order']

    ax.plot(data, 'kx', label='data', markersize=3, alpha=0.5)
    ax.plot(fundamental_signal, '-', color=[0.5, 0.5, 0.5], label='fundamental_signal', linewidth=1.5)

    # Display range: first 3 periods or at least 100 points
    xlim_max = min(max(int(3 / normalized_freq), 100), n_samples)
    ax.set_xlim([0, xlim_max])
    data_min, data_max = np.min(data), np.max(data)
    ax.set_ylim([data_min * 1.1, data_max * 1.1])
    ax.set_ylabel('Signal', color='k')
    ax.tick_params(axis='y', labelcolor='k')

    # Right Y-axis for errors
    ax2 = ax.twinx()

    # Calculate RMS and percentages
    rms = np.sqrt(np.mean(np.array([harmonic_error, other_error, total_error])**2, axis=1))
    rms_harmonic, rms_other, rms_total = rms
    pct_harmonic = (rms_harmonic / rms_total)**2 * 100
    pct_other = (rms_other / rms_total)**2 * 100

    # Select unit based on magnitude
    scale, unit = (1e6, 'uV') if rms_total < 1e-3 else (1e3, 'mV') if rms_total < 1 else (1, 'V')

    ax2.plot(harmonic_error, 'r-', label=f'harmonics ({rms_harmonic*scale:.1f}{unit}, {pct_harmonic:.1f}%)', linewidth=1.5)
    ax2.plot(other_error, 'b-', label=f'other errors ({rms_other*scale:.1f}{unit}, {pct_other:.1f}%)', linewidth=1)

    error_min, error_max = np.min(total_error), np.max(total_error)
    ax2.set_ylim([error_min * 1.1, error_max * 1.1])
    ax2.set_ylabel('Error', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    ax.set_xlabel('Samples')
    ax.set_title(f'Decompose Harmonics (freq={normalized_freq:.6f}, order={order})')

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax.grid(True, alpha=0.3)
