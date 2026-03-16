"""Bit weight and radix visualization for ADC calibration analysis.

Visualizes absolute bit weights with radix annotations to identify scaling patterns.
Computes weight scaling factor (wgtsca) and effective resolution (effres).
"""

import numpy as np
import matplotlib.pyplot as plt

def analyze_weight_radix(
    weights: np.ndarray,
    create_plot: bool = True,
    ax=None,
    title: str | None = None
) -> dict:
    """
    Visualize absolute bit weights with radix annotations.

    Pure binary: radix = 2.00. Sub-radix/redundancy: radix < 2.00.

    Parameters
    ----------
    weights : np.ndarray
        Bit weights (1D array), from MSB to LSB
    create_plot : bool, default=True
        If True, create line plot with radix annotations
    ax : plt.Axes, optional
        Axes to plot on. If None, uses current axes (plt.gca())
    title : str, optional
        Title for the plot. If None, uses default title

    Returns
    -------
    dict
        'radix': Radix between consecutive bits (weight[i-1]/weight[i])
        'wgtsca': Optimal weight scaling factor
        'effres': Effective resolution in bits

    Notes
    -----
    What to look for in radix values:
    - Radix = 2.00: Binary scaling (SAR, pure binary)
    - Radix < 2.00: Redundancy or sub-radix (e.g., 1.5-bit/stage → ~1.90)
    - Radix > 2.00: Unusual, may indicate calibration error
    - Consistent pattern: Expected architecture behavior
    - Random jumps: Calibration errors or bit mismatch
    """
    weights = np.asarray(weights, dtype=float)
    n_bits = len(weights)
    is_negative = weights < 0
    abs_weights = np.abs(weights)

    # Calculate radix between consecutive bits (|weight[i-1]| / |weight[i]|)
    radix = np.zeros(n_bits)
    radix[0] = np.nan  # No radix for first bit
    for i in range(1, n_bits):
        radix[i] = abs_weights[i-1] / abs_weights[i]

    # --- Compute wgtsca and effres ---
    # Step 1: Sort absolute weights descending
    sort_idx = np.argsort(abs_weights)[::-1]
    abs_w_sorted = abs_weights[sort_idx]

    # Step 2: Calculate ratios between adjacent sorted weights
    ratios = abs_w_sorted[:-1] / abs_w_sorted[1:]

    # Step 3: Find significance threshold (first ratio >= 3.0)
    sig_break = np.where(ratios >= 3.0)[0]
    if len(sig_break) == 0:
        k = len(ratios)  # all bits significant
    else:
        k = sig_break[0]  # k ratios are < 3, so k+1 bits significant

    # Track MSB/LSB indices (in original array order)
    msb_idx = sort_idx[0]        # index of largest weight
    lsb_idx = sort_idx[k]        # index of smallest significant weight

    # Significant weights (sorted descending)
    abs_w_sig = abs_w_sorted[:k + 1]

    # Step 4: Initial wgtsca - normalize smallest significant weight to 1
    wgtsca = 1.0 / abs_w_sig[-1]

    # Step 5: Refine wgtsca by searching integer MSB values
    w_err = np.sqrt(np.mean((abs_w_sig * wgtsca - np.round(abs_w_sig * wgtsca)) ** 2))
    wmsb_init = round(abs_w_sorted[0] * wgtsca)
    wmsb_min = max(1, round(wmsb_init * 0.5))
    wmsb_max = max(wmsb_min, round(wmsb_init * 1.5))
    for wmsb in range(wmsb_min, wmsb_max + 1):
        w_refine = wmsb / abs_w_sorted[0]
        w_err_ref = np.sqrt(np.mean((abs_w_sig * w_refine - np.round(abs_w_sig * w_refine)) ** 2))
        if w_err_ref < w_err:
            w_err = w_err_ref
            wgtsca = w_refine

    # Step 6: Effective resolution
    effres = np.log2(np.sum(abs_w_sig) / abs_w_sig[-1] + 1)

    if create_plot:
        if ax is None:
            ax = plt.gca()

        # Plot connecting line (black)
        ax.plot(range(1, n_bits + 1), abs_weights, '-', linewidth=2,
                color=[0, 0, 0], label='_nolegend_')

        # Positive weight markers (blue)
        pos_idx = np.where(~is_negative)[0]
        h_pos = None
        if len(pos_idx) > 0:
            h_pos = ax.plot(pos_idx + 1, abs_weights[pos_idx], 'o', markersize=8,
                            markerfacecolor=(0.3, 0.6, 0.8), color=(0.3, 0.6, 0.8),
                            linewidth=2, label='Positive')[0]

        # Negative weight markers (red)
        neg_idx = np.where(is_negative)[0]
        h_neg = None
        if len(neg_idx) > 0:
            h_neg = ax.plot(neg_idx + 1, abs_weights[neg_idx], 'o', markersize=8,
                            markerfacecolor=(0.9, 0.3, 0.3), color=(0.9, 0.3, 0.3),
                            linewidth=2, label='Negative')[0]

        ax.set_xlabel('Bit Index', fontsize=14)
        ax.set_ylabel('Absolute Weight', fontsize=14)

        if title is not None:
            ax.set_title(title, fontsize=16)
        else:
            ax.set_title('Bit Weights with Radix', fontsize=16)

        ax.grid(True)
        ax.set_xlim([0.5, n_bits + 0.5])
        ax.tick_params(labelsize=14)
        ax.set_yscale('log')

        # MSB label
        ax.text(msb_idx + 1, abs_weights[msb_idx] / 1.5, 'MSB',
                ha='center', va='top', fontsize=10,
                color=(0.8, 0.1, 0.1), fontweight='bold')

        # LSB label (only if different from MSB)
        if lsb_idx != msb_idx:
            ax.text(lsb_idx + 1, abs_weights[lsb_idx] / 1.5, 'LSB',
                    ha='center', va='top', fontsize=10,
                    color=(0.1, 0.5, 0.1), fontweight='bold')

        # Effective resolution annotation (upper-right)
        ax.text(0.98, 0.88, f'Eff. Res: {effres:.2f} bits',
                transform=ax.transAxes, ha='right', va='top',
                fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=(0.5, 0.5, 0.5)))

        # Legend when negative weights present
        if h_pos is not None and h_neg is not None:
            ax.legend(loc='best')
        elif h_neg is not None:
            ax.legend(loc='best')

        # Annotate radix on top of each data point (except first bit)
        for b in range(1, n_bits):
            y_pos = abs_weights[b] * 1.5
            ax.text(b + 1, y_pos, f'/{radix[b]:.2f}',
                    ha='center', fontsize=10, color=[0.2, 0.2, 0.2], fontweight='bold')

    return {'radix': radix, 'wgtsca': wgtsca, 'effres': effres}
