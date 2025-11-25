"""
Overflow Check Tool for SAR ADC

Analyzes residue distribution at each bit position to detect overflow conditions.
This is useful for sub-radix-2 SAR ADC calibration and redundancy analysis.

Ported from MATLAB: overflowChk.m
"""

import numpy as np
import matplotlib.pyplot as plt


def overflow_chk(raw_code, weight, OFB=None):
    """
    Analyze residue distribution at each bit position (matching MATLAB exactly).

    For each bit position, calculates the normalized residue (remaining bits weighted sum).
    Detects overflow conditions where residue goes outside [0, 1] range.

    Parameters
    ----------
    raw_code : ndarray
        Digital codes array, shape (N, M) where N=samples, M=bits.
        Each row is one sample, each column is one bit (MSB first).
    weight : ndarray
        Weight array for each bit, shape (M,).
    OFB : int, optional
        Overflow bit position for overflow detection.
        Default is M (LSB position, 0-indexed from MSB).

    Returns
    -------
    None
        This function creates a plot using matplotlib.
    """
    raw_code = np.asarray(raw_code)
    weight = np.asarray(weight)

    if raw_code.ndim == 1:
        raw_code = raw_code.reshape(-1, 1)

    N, M = raw_code.shape

    if len(weight) != M:
        raise ValueError(f"Weight length ({len(weight)}) must match number of bits ({M})")

    # Default OFB is M (LSB)
    if OFB is None:
        OFB = M

    data_decom = np.zeros((N, M))
    range_min = np.zeros(M)
    range_max = np.zeros(M)

    # Calculate normalized residue at each bit position
    for ii in range(M):
        # Weighted sum of remaining bits (from current bit to LSB)
        tmp = raw_code[:, ii:] @ weight[ii:]

        # Normalize by sum of remaining weights
        sum_weight = np.sum(weight[ii:])
        data_decom[:, ii] = tmp / sum_weight
        range_min[ii] = np.min(tmp) / sum_weight
        range_max[ii] = np.max(tmp) / sum_weight

    # Detect overflow at specified bit position
    # MATLAB: ovf_zero = (data_decom(:,M-OFB+1) <= 0);
    # Python 0-indexed: M-OFB+1-1 = M-OFB
    ovf_zero = data_decom[:, M - OFB] <= 0
    ovf_one = data_decom[:, M - OFB] >= 1
    non_ovf = ~(ovf_zero | ovf_one)

    # Create plot matching MATLAB style
    fig = plt.gcf()
    if fig is None or len(fig.get_axes()) == 0:
        fig = plt.figure(figsize=(10, 6))

    ax = plt.gca()

    # Hold on for multiple plots
    ax.hold = True

    # Reference lines at 0 and 1
    ax.plot([0, M + 1], [1, 1], '-k', linewidth=1)
    ax.plot([0, M + 1], [0, 0], '-k', linewidth=1)

    # Plot min/max range envelope
    bit_positions = np.arange(1, M + 1)
    ax.plot(bit_positions, range_min, '-r', linewidth=1)
    ax.plot(bit_positions, range_max, '-r', linewidth=1)

    # Scatter plot for each bit position
    for ii in range(M):
        # All samples (blue) with very low alpha
        h = ax.scatter(
            np.ones(N) * (ii + 1),
            data_decom[:, ii],
            c='b',
            s=8,
            alpha=0.01,
            edgecolors='none'
        )

        # Overflow high samples (red, shifted left)
        n_ovf_one = np.sum(ovf_one)
        if n_ovf_one > 0:
            h = ax.scatter(
                np.ones(n_ovf_one) * (ii + 1) - 0.2,
                data_decom[ovf_one, ii],
                c='r',
                s=8,
                alpha=0.01,
                edgecolors='none'
            )

        # Overflow low samples (yellow, shifted right)
        n_ovf_zero = np.sum(ovf_zero)
        if n_ovf_zero > 0:
            h = ax.scatter(
                np.ones(n_ovf_zero) * (ii + 1) + 0.2,
                data_decom[ovf_zero, ii],
                c='y',
                s=8,
                alpha=0.01,
                edgecolors='none'
            )

        # Percentage labels (matching MATLAB format)
        ovf_zero_pct = np.sum(data_decom[:, ii] <= 0) / N * 100
        ovf_one_pct = np.sum(data_decom[:, ii] >= 1) / N * 100

        ax.text(ii + 1, -0.05, f'{ovf_zero_pct:.1f}%', ha='center', va='top', fontsize=9)
        ax.text(ii + 1, 1.05, f'{ovf_one_pct:.1f}%', ha='center', va='bottom', fontsize=9)

    # Set axis limits and labels (matching MATLAB)
    ax.set_xlim([0, M + 1])
    ax.set_ylim([-0.1, 1.1])

    # X-axis ticks: bit positions from MSB to LSB
    ax.set_xticks(bit_positions)
    ax.set_xticklabels([str(M - i) for i in range(M)])

    ax.set_xlabel('bit')
    ax.set_ylabel('Residue Distribution')

    # Note: Title is set externally in the test script
