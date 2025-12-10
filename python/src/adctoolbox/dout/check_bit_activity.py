"""Analyze and plot the percentage of 1's in each bit."""

import numpy as np
import matplotlib.pyplot as plt


def check_bit_activity(bits):
    """
    Analyze and plot the percentage of 1's in each bit.

    Parameters
    ----------
    bits : array_like
        Binary matrix (N x B), N=samples, B=bits (MSB to LSB)

    Returns
    -------
    bit_usage : ndarray
        Percentage of 1's for each bit (1D array of length B)

    Example
    -------
    >>> bits = np.loadtxt('dout_SAR_12b.csv', delimiter=',')
    >>> bit_usage = bit_activity(bits)
    """
    bits = np.asarray(bits)
    B = bits.shape[1]
    bit_usage = np.mean(bits, axis=0) * 100

    bars = plt.bar(range(1, B+1), bit_usage, color='steelblue', edgecolor='black', linewidth=0.5)

    max_dev_idx = np.argmax(np.abs(bit_usage - 50))
    max_dev_value = bit_usage[max_dev_idx]
    bars[max_dev_idx].set_color('orange')
    bars[max_dev_idx].set_edgecolor('red')
    bars[max_dev_idx].set_linewidth(2)

    plt.text(max_dev_idx + 1, max_dev_value + 0.8, f'{max_dev_value - 50:+.2f}%',
             ha='center', va='bottom', fontsize=10, fontweight='bold', color='red')

    plt.axhline(50, color='red', linestyle='--', linewidth=2, label='Ideal (50%)')
    plt.xlim([0.5, B + 0.5])
    plt.ylim([40, 60])
    plt.xlabel('Bit Index (1=MSB)', fontsize=11)
    plt.ylabel("Activity of 1's (%)", fontsize=11)
    plt.title('Bit Activity', fontsize=11, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3, axis='y')

    return bit_usage
