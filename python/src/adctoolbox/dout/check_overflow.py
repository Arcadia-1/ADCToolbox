"""
Overflow Check Tool for SAR ADC

Analyzes residue distribution at each bit position to detect overflow conditions.
This is useful for sub-radix-2 SAR ADC calibration and redundancy analysis.

Ported from MATLAB: overflowChk.m
"""

import numpy as np
import matplotlib.pyplot as plt


def check_overflow(raw_code, weight, ofb=None, disp=False):
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
    ofb : int, optional
        Overflow bit position for overflow detection.
        Default is M (check at MSB, MATLAB convention: 1=LSB, M=MSB).
    disp : bool, optional
        Display plot (default: False). Set to True to generate visualization.

    Returns
    -------
    range_min : ndarray
        Minimum normalized residue for each bit position, shape (M,).
        Shows how close each bit segment gets to underflow (0).
    range_max : ndarray
        Maximum normalized residue for each bit position, shape (M,).
        Shows how close each bit segment gets to overflow (1).
    ovf_percent_zero : ndarray
        Percentage of samples at or below 0 for each bit, shape (M,).
        Underflow percentage per bit position.
    ovf_percent_one : ndarray
        Percentage of samples at or above 1 for each bit, shape (M,).
        Overflow percentage per bit position.

    Examples
    --------
    >>> bits = np.random.randint(0, 2, size=(10000, 12))
    >>> weight = 2**np.arange(11, -1, -1)
    >>> range_min, range_max, pct_zero, pct_one = overflow_chk(bits, weight)

    Notes
    -----
    - A bit segment is the sub-code formed from one bit to the LSB
    - Residue is normalized by dividing by the sum of weights in the corresponding segment
    - Matches MATLAB ovfchk.m behavior exactly
    """
    raw_code = np.asarray(raw_code)
    weight = np.asarray(weight)

    if raw_code.ndim == 1:
        raw_code = raw_code.reshape(-1, 1)

    N, M = raw_code.shape

    if len(weight) != M:
        raise ValueError(f"Weight length ({len(weight)}) must match number of bits ({M})")

    # Default ofb is M (LSB)
    if ofb is None:
        ofb = M

    data_decom = np.zeros((N, M))
    range_min = np.zeros(M)
    range_max = np.zeros(M)
    ovf_percent_zero = np.zeros(M)
    ovf_percent_one = np.zeros(M)

    # Calculate normalized residue at each bit position
    for ii in range(M):
        # Weighted sum of remaining bits (from current bit to LSB)
        tmp = raw_code[:, ii:] @ weight[ii:]

        # Normalize by sum of remaining weights
        sum_weight = np.sum(weight[ii:])
        data_decom[:, ii] = tmp / sum_weight
        range_min[ii] = np.min(tmp) / sum_weight
        range_max[ii] = np.max(tmp) / sum_weight

        # Calculate overflow percentages
        ovf_percent_zero[ii] = np.sum(data_decom[:, ii] <= 0) / N * 100
        ovf_percent_one[ii] = np.sum(data_decom[:, ii] >= 1) / N * 100

    # Detect overflow at specified bit position
    # MATLAB: ovf_zero = (data_decom(:,M-chkpos+1) <= 0);
    # Python 0-indexed: M-ofb+1-1 = M-ofb
    ovf_zero = data_decom[:, M - ofb] <= 0
    ovf_one = data_decom[:, M - ofb] >= 1
    non_ovf = ~(ovf_zero | ovf_one)

    # Only plot if display requested
    if disp:
        # Create plot matching MATLAB style
        fig = plt.gcf()
        if fig is None or len(fig.get_axes()) == 0:
            fig = plt.figure(figsize=(10, 6))

        ax = plt.gca()

        # Reference lines at 0 and 1 (matching MATLAB)
        ax.plot([0, M + 1], [1, 1], '-k', linewidth=0.5)
        ax.plot([0, M + 1], [0, 0], '-k', linewidth=0.5)

        # Plot min/max range envelope (matching MATLAB)
        bit_positions = np.arange(1, M + 1)
        ax.plot(bit_positions, range_min, '-r', linewidth=1.5)
        ax.plot(bit_positions, range_max, '-r', linewidth=1.5)

        # Scatter plot for each bit position (matching MATLAB style)
        for ii in range(M):
            # Normal samples (blue)
            alpha = min(max(10 / N, 0.01), 1)
            if np.sum(non_ovf) > 0:
                ax.scatter(
                    np.ones(np.sum(non_ovf)) * (ii + 1),
                    data_decom[non_ovf, ii],
                    s=36,
                    facecolors='b',
                    edgecolors='b',
                    alpha=alpha,
                    linewidths=0.5
                )

            # Overflow high samples (red, shifted left)
            if np.sum(ovf_one) > 0:
                ax.scatter(
                    np.ones(np.sum(ovf_one)) * (ii + 1) - 0.2,
                    data_decom[ovf_one, ii],
                    s=36,
                    facecolors='r',
                    edgecolors='r',
                    alpha=alpha,
                    linewidths=0.5
                )

            # Overflow low samples (yellow, shifted right)
            if np.sum(ovf_zero) > 0:
                ax.scatter(
                    np.ones(np.sum(ovf_zero)) * (ii + 1) + 0.2,
                    data_decom[ovf_zero, ii],
                    s=36,
                    facecolors='y',
                    edgecolors='y',
                    alpha=alpha,
                    linewidths=0.5
                )

            # Percentage labels (matching MATLAB format)
            ax.text(ii + 1, -0.05, f'{ovf_percent_zero[ii]:.1f}%', ha='center', va='top', fontsize=10)
            ax.text(ii + 1, 1.05, f'{ovf_percent_one[ii]:.1f}%', ha='center', va='bottom', fontsize=10)

        # Set axis limits and labels (matching MATLAB)
        ax.set_xlim([0, M + 1])
        ax.set_ylim([-0.1, 1.1])

        # X-axis ticks: bit positions from MSB to LSB
        ax.set_xticks(bit_positions)
        ax.set_xticklabels([str(M - i) for i in range(M)])

        ax.set_xlabel('bit')
        ax.set_ylabel('Residue Distribution')

        # Note: Title is set externally in the test script

    return range_min, range_max, ovf_percent_zero, ovf_percent_one
