"""Bit weight and radix visualization for ADC calibration analysis.

Visualizes absolute bit weights with radix annotations to identify scaling patterns.
"""

import numpy as np
import matplotlib.pyplot as plt

def analyze_weight_radix(
    weights: np.ndarray,
    create_plot: bool = True,
    ax=None,
    title: str | None = None
) -> np.ndarray:
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
    np.ndarray
        Radix between consecutive bits (weight[i-1]/weight[i])

    Notes
    -----
    What to look for in radix values:
    - Radix = 2.00: Binary scaling (SAR, pure binary)
    - Radix < 2.00: Redundancy or sub-radix (e.g., 1.5-bit/stage → ~1.90)
    - Radix > 2.00: Unusual, may indicate calibration error
    - Consistent pattern: Expected architecture behavior
    - Random jumps: Calibration errors or bit mismatch
    """
    weights = np.asarray(weights)
    n_bits = len(weights)

    # Calculate radix between consecutive bits (weight[i-1] / weight[i])
    radix = np.zeros(n_bits)
    radix[0] = np.nan  # No radix for first bit
    for i in range(1, n_bits):
        radix[i] = weights[i-1] / weights[i]

    if create_plot:
        if ax is None:
            ax = plt.gca()

        # Create line plot with markers showing absolute weights
        ax.plot(range(1, n_bits + 1), weights, '-o', linewidth=2, markersize=8,
                markerfacecolor=[0.3, 0.6, 0.8], color=[0.3, 0.6, 0.8])
        ax.set_xlabel('Bit Index (1=MSB, N=LSB)', fontsize=14)
        ax.set_ylabel('Absolute Weight', fontsize=14)

        # Set title if provided
        if title is not None:
            ax.set_title(title, fontsize=16)
        else:
            ax.set_title('Bit Weights with Radix', fontsize=16)

        ax.grid(True)
        ax.set_xlim([0.5, n_bits + 0.5])
        ax.tick_params(labelsize=14)
        ax.set_yscale('log')  # Log scale for better visualization

        # Annotate radix on top of each data point (except first bit)
        for b in range(1, n_bits):
            y_pos = weights[b] * 1.5  # Position text above the marker
            ax.text(b + 1, y_pos, f'/{radix[b]:.2f}',
                    ha='center', fontsize=10, color=[0.2, 0.2, 0.2], fontweight='bold')

    return radix
