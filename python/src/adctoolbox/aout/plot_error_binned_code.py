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
        Dictionary from compute_error_by_code(). Must contain:
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
    >>> from adctoolbox.aout import compute_error_by_code
    >>> sig = np.sin(2*np.pi*0.1*np.arange(1000))
    >>> results = compute_error_by_code(sig, 0.1, num_bits=10)
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

    # --- Top Panel: Mean Error vs Code (INL-like) ---
    if np.any(valid_mask):
        ax1.plot(code_bins[valid_mask], emean_by_code[valid_mask],
                'b-', linewidth=1, label='Mean Error')
        ax1.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

        # Add ±1 reference lines if error range is small
        emean_range = np.nanmax(np.abs(emean_by_code))
        if emean_range > 0 and emean_range < 10:
            ax1.axhline(1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
            ax1.axhline(-1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    ax1.set_xlim([code_min, code_max])
    ax1.set_ylabel('Mean Error (INL-like)')
    ax1.set_title('Code Error Analysis: Mean Error vs Code')
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
    if np.any(valid_mask):
        ax2.plot(code_bins[valid_mask], erms_by_code[valid_mask],
                'r-', linewidth=1, label='RMS Error')

        # Optionally show bin counts as background
        ax2_twin = ax2.twinx()
        ax2_twin.fill_between(code_bins, 0, bin_counts, alpha=0.2, color='gray',
                             label='Sample Count')
        ax2_twin.set_ylabel('Sample Count', color='gray')
        ax2_twin.tick_params(axis='y', labelcolor='gray')

    ax2.set_xlim([code_min, code_max])
    ax2.set_ylim([0, np.nanmax(erms_by_code)*1.2 if np.any(valid_mask) else 1.0])
    ax2.set_xlabel('Code')
    ax2.set_ylabel('RMS Error', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Set x-axis ticks if num_bits provided
    if num_bits is not None:
        full_scale = 2**num_bits
        tick_positions = [full_scale * i / 8 for i in range(9)]
        tick_labels = [f'{int(pos)}' for pos in tick_positions]
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels(tick_labels)

    # Add statistics annotations
    if np.any(valid_mask):
        max_erms = np.nanmax(erms_by_code)
        mean_erms = np.nanmean(erms_by_code)
        max_emean = np.nanmax(np.abs(emean_by_code))

        text_y = max_erms * 1.15 if max_erms > 0 else 1.0
        ax2.text(code_min + (code_max - code_min) * 0.02, text_y,
                f'Max |Mean Error| = {max_emean:.3e}  |  Mean RMS = {mean_erms:.3e}',
                color='k', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax2.set_title('RMS Error vs Code (Code-Dependent Noise)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')

    plt.tight_layout()
