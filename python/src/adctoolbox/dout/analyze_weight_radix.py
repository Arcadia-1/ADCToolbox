"""Visualize absolute bit weights with radix annotations."""

import numpy as np
import matplotlib.pyplot as plt


def plot_weight_radix(weights):
    """
    Visualize absolute bit weights with radix annotations.

    Parameters
    ----------
    weights : array_like
        Bit weights (1D array), from MSB to LSB

    Returns
    -------
    radix : ndarray
        Radix between consecutive bits (weight[i-1]/weight[i])

    Description
    -----------
    Visualizes absolute bit weight values with annotations showing the
    radix (scaling factor) relative to the previous bit.

    Pure binary weights have radix = 2.00 (each bit is half the previous).
    Architectures with redundancy or sub-radix may deviate from 2.00.

    What to look for:
    - Radix = 2.00: Binary scaling (SAR, pure binary)
    - Radix < 2.00: Redundancy or sub-radix (e.g., 1.5-bit/stage → ~1.90)
    - Radix > 2.00: Unusual, may indicate calibration error
    - Consistent pattern: Expected architecture behavior
    - Random jumps: Calibration errors or bit mismatch

    Example
    -------
    >>> bits = np.loadtxt('dout_SAR_12b.csv', delimiter=',')
    >>> weight_cal, _, _, _, _, _ = cal_weight_sine(bits, freq=0, order=5)
    >>> radix = plot_weight_radix(weight_cal)
    """
    weights = np.asarray(weights)
    n_bits = len(weights)

    # Calculate radix between consecutive bits (weight[i-1] / weight[i])
    radix = np.zeros(n_bits)
    radix[0] = np.nan  # No radix for first bit
    for i in range(1, n_bits):
        radix[i] = weights[i-1] / weights[i]

    # Create line plot with markers showing absolute weights
    plt.plot(range(1, n_bits + 1), weights, '-o', linewidth=2, markersize=8,
             markerfacecolor=[0.3, 0.6, 0.8], color=[0.3, 0.6, 0.8])
    plt.xlabel('Bit Index (1=MSB, N=LSB)', fontsize=14)
    plt.ylabel('Absolute Weight', fontsize=14)
    plt.title('Bit Weights with Radix', fontsize=16)
    plt.grid(True)
    plt.xlim([0.5, n_bits + 0.5])
    plt.gca().tick_params(labelsize=14)
    plt.yscale('log')  # Log scale for better visualization

    # Annotate radix on top of each data point (except first bit)
    for b in range(1, n_bits):
        y_pos = weights[b] * 1.5  # Position text above the marker
        plt.text(b + 1, y_pos, f'/{radix[b]:.2f}',
                ha='center', fontsize=10, color=[0.2, 0.2, 0.2], fontweight='bold')

    return radix
