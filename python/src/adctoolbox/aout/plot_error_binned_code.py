"""
Plot code-based error analysis (INL-like curves).

Visualization function for displaying code-binned error analysis results,
showing mean and RMS error as a function of ADC code.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_error_binned_code(results: dict, ax=None):
    """
    Plot code-based error analysis (INL-like curves).

    Creates a comprehensive visualization showing:
    - Top panel: Mean error vs code (INL-like)
    - Bottom panel: RMS error vs code (code-dependent noise)

    Parameters
    ----------
    results : dict
        Dictionary from rearrange_error_by_code(). Must contain:
        - 'emean_by_code': Mean error per code bin
        - 'erms_by_code': RMS error per code bin
        - 'code_bins': Code bin centers
        - 'bin_counts': Number of samples per bin
        - 'num_bits': Number of bits (optional)
        - 'code_min': Minimum code value
        - 'code_max': Maximum code value
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure with 2 subplots.

    Notes
    -----
    The top panel shows mean error vs code, which reveals:
    - Static nonlinearity (INL-like patterns)
    - Systematic code-dependent errors
    - Missing codes (gaps in data)

    The bottom panel shows RMS error vs code, which reveals:
    - Code-dependent noise
    - Quantization effects
    - Non-uniform noise distribution

    Examples
    --------
    >>> from adctoolbox.aout import rearrange_error_by_code
    >>> sig = np.sin(2*np.pi*0.1*np.arange(1000))
    >>> results = rearrange_error_by_code(sig, 0.1, num_bits=10)
    >>> plot_error_binned_code(results)
    """
    # Extract data from results
    emean_by_code = results['emean_by_code']
    erms_by_code = results['erms_by_code']
    code_bins = results['code_bins']
    bin_counts = results['bin_counts']
    num_bits = results.get('num_bits', None)
    code_min = results['code_min']
    code_max = results['code_max']

    # Extract raw error and code data for scatter plot (matching MATLAB errsin.m style)
    error_raw = results.get('error', None)
    codes_raw = results.get('codes', None)

    # Create figure if no axes provided
    if ax is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    else:
        # Split provided axis into 2 subplots
        fig = ax.get_figure()
        pos = ax.get_position()
        ax.remove()
        ax1 = fig.add_axes([pos.x0, pos.y0 + pos.height/2, pos.width, pos.height/2])
        ax2 = fig.add_axes([pos.x0, pos.y0, pos.width, pos.height/2])

    # Filter valid data (non-NaN)
    valid_mask = ~np.isnan(emean_by_code)

    # --- Top Panel: Error vs Code (INL-like) ---
    # Following MATLAB errsin.m style (lines 153-164):
    # - Scatter plot of individual errors (plot(sig, err, 'r.'))
    # - Overlay line plot of mean errors (plot(xx, emean, 'b-'))

    # Plot scatter of raw errors if available
    if error_raw is not None and codes_raw is not None:
        ax1.plot(codes_raw, error_raw, 'r.', markersize=3, alpha=0.6, label='Error')

    # Plot mean error as line overlaid on scatter
    if np.any(valid_mask):
        ax1.plot(code_bins[valid_mask], emean_by_code[valid_mask],
                'b-', linewidth=2, label='Mean Error')

    ax1.set_xlim([code_min, code_max])
    ax1.set_ylabel('Error')
    ax1.set_xlabel('Value')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

    # Set x-axis ticks if num_bits provided
    if num_bits is not None:
        full_scale = 2**num_bits
        tick_positions = [full_scale * i / 8 for i in range(9)]
        tick_labels = [f'{int(pos)}' for pos in tick_positions]
        ax1.set_xticks(tick_positions)
        ax1.set_xticklabels(tick_labels)

    # --- Bottom Panel: RMS Error vs Code ---
    # Following MATLAB errsin.m style: bar plot of RMS errors
    if np.any(valid_mask):
        bin_width = np.mean(np.diff(code_bins[valid_mask])) if len(code_bins[valid_mask]) > 1 else 1
        ax2.bar(code_bins[valid_mask], erms_by_code[valid_mask],
               width=bin_width*0.8, color='steelblue', alpha=0.7, label='RMS Error')

    ax2.set_xlim([code_min, code_max])
    ax2.set_ylim([0, np.nanmax(erms_by_code)*1.1 if np.any(valid_mask) else 1.0])
    ax2.set_xlabel('Code')
    ax2.set_ylabel('RMS Error')

    # Set x-axis ticks if num_bits provided
    if num_bits is not None:
        full_scale = 2**num_bits
        tick_positions = [full_scale * i / 8 for i in range(9)]
        tick_labels = [f'{int(pos)}' for pos in tick_positions]
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels(tick_labels)

    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    plt.tight_layout()
