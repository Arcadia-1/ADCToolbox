"""Analyze and plot the percentage of 1's in each bit."""

import numpy as np
import matplotlib.pyplot as plt


def bit_activity(bits, annotate_extremes=True):
    """
    Analyze and plot the percentage of 1's in each bit.

    Parameters
    ----------
    bits : array_like
        Binary matrix (N x B), N=samples, B=bits (MSB to LSB)
    annotate_extremes : bool, optional
        Annotate bits with >95% or <5% activity (default: True)

    Returns
    -------
    bit_usage : ndarray
        Percentage of 1's for each bit (1D array of length B)

    Description
    -----------
    This function calculates and visualizes the percentage of 1's in each bit
    position. A bar chart is created with a reference line at 50% (ideal).

    What to look for:
    - ~50%: Good bit activity, well-utilized
    - >95%: Bit stuck high or large positive DC offset
    - <5%:  Bit stuck low or large negative DC offset
    - Gradual trend: Indicates DC offset pattern across MSB→LSB

    Example
    -------
    >>> bits = np.loadtxt('dout_SAR_12b.csv', delimiter=',')
    >>> bit_usage = bit_activity(bits)
    """
    bits = np.asarray(bits)

    # Calculate percentage of 1's for each bit
    n_bits = bits.shape[1]
    bit_usage = np.mean(bits, axis=0) * 100  # Percentage of 1's per bit

    # Create bar chart
    plt.bar(range(1, n_bits + 1), bit_usage, color=[0.2, 0.4, 0.8])
    plt.axhline(50, color='r', linestyle='--', linewidth=2, label='Ideal (50%)')
    plt.xlabel('Bit Index (1=MSB, N=LSB)')
    plt.ylabel("Percentage of 1's (%)")
    plt.title('Bit Activity Analysis')
    plt.ylim([0, 100])
    plt.xlim([0.5, n_bits + 0.5])
    plt.grid(True)
    plt.legend()

    # Add text annotations for extreme values
    if annotate_extremes:
        for b in range(n_bits):
            if bit_usage[b] > 95:
                plt.text(b + 1, bit_usage[b] + 3, f'{bit_usage[b]:.1f}%',
                        ha='center', fontsize=10, color='red', fontweight='bold')
            elif bit_usage[b] < 5:
                plt.text(b + 1, bit_usage[b] + 3, f'{bit_usage[b]:.1f}%',
                        ha='center', fontsize=10, color='red', fontweight='bold')

    return bit_usage
