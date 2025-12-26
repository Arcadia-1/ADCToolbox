"""Bit activity analysis for SAR ADC digital outputs.

Analyzes the percentage of 1's in each bit to detect DC offset and clipping.
"""

import numpy as np
import matplotlib.pyplot as plt

def analyze_bit_activity(
    bits: np.ndarray,
    create_plot: bool = True,
    ax=None,
    title: str | None = None
) -> np.ndarray:
    """
    Analyze and plot the percentage of 1's in each bit.

    Ideal SAR ADC should have 50% activity for all bits. Deviations indicate
    input signal DC offset or amplitude clipping.

    Parameters
    ----------
    bits : np.ndarray
        Binary matrix (N x B), N=samples, B=bits (MSB to LSB)
    create_plot : bool, default=True
        If True, create bar chart visualization
    ax : plt.Axes, optional
        Axes to plot on. If None, uses current axes (plt.gca())
    title : str, optional
        Title for the plot. If None, uses default title

    Returns
    -------
    np.ndarray
        Percentage of 1's for each bit (1D array of length B)
    """
    bits = np.asarray(bits)
    B = bits.shape[1]
    bit_usage = np.mean(bits, axis=0) * 100

    if create_plot:
        if ax is None:
            ax = plt.gca()

        bars = ax.bar(range(1, B+1), bit_usage, color='steelblue', edgecolor='black', linewidth=0.5)

        max_dev_idx = np.argmax(np.abs(bit_usage - 50))
        max_dev_value = bit_usage[max_dev_idx]
        bars[max_dev_idx].set_color('orange')
        bars[max_dev_idx].set_edgecolor('red')
        bars[max_dev_idx].set_linewidth(2)

        ax.text(max_dev_idx + 1, max_dev_value + 0.8, f'{max_dev_value - 50:+.2f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='red')

        ax.axhline(50, color='red', linestyle='--', linewidth=2, label='Ideal (50%)')
        ax.set_xlim([0.5, B + 0.5])
        ax.set_ylim([40, 60])
        ax.set_xlabel('Bit Index (1=MSB)', fontsize=11)
        ax.set_ylabel("Activity of 1's (%)", fontsize=11)

        # Set title if provided
        if title is not None:
            ax.set_title(title, fontsize=11, fontweight='bold')
        else:
            ax.set_title('Bit Activity', fontsize=11, fontweight='bold')

        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    return bit_usage
