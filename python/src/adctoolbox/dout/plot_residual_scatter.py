"""Residual scatter plot for ADC bit-stage analysis.

Visualizes partial-sum residuals between bit stages to reveal
correlations, nonlinearity patterns, and redundancy.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_residual_scatter(
    signal: np.ndarray,
    bits: np.ndarray,
    weights: np.ndarray | None = None,
    pairs: list[tuple[int, int]] | None = None,
    alpha: float | str = 'auto',
    create_plot: bool = True,
) -> dict:
    """
    Plot partial-sum residuals of an ADC bit matrix.

    For each pair (x_bit, y_bit), computes the residual after subtracting
    the first x_bit (or y_bit) weighted bits from the input signal, then
    plots y-residual vs x-residual as a scatter plot.

    Parameters
    ----------
    signal : np.ndarray
        Ideal input signal to the ADC (1D, length N).
    bits : np.ndarray
        Raw ADC output bit matrix (N x M), MSB-first columns.
    weights : np.ndarray, optional
        Bit weights (length M). Default: binary [2^(M-1), ..., 1].
    pairs : list of (int, int), optional
        Pairs of bit indices whose residuals are plotted.
        Range: 0 (raw signal) to M (residual after all bits).
        Default: [(0, M), (1, M), ..., (M-1, M)].
    alpha : float or 'auto', default='auto'
        Marker transparency. 'auto' scales as clamp(1000/N, 0.1, 1.0).
    create_plot : bool, default=True
        If True, create scatter subplots.

    Returns
    -------
    dict
        'pairs': list of (x_bit, y_bit) tuples used
        'residuals_x': list of 1D arrays, one per pair
        'residuals_y': list of 1D arrays, one per pair
    """
    signal = np.asarray(signal, dtype=float).ravel()
    bits = np.asarray(bits, dtype=float)
    n, m = bits.shape

    if len(signal) != n:
        raise ValueError(
            f"signal length ({len(signal)}) must match bits rows ({n})."
        )

    # Default weights: binary
    if weights is None:
        weights = np.power(2.0, np.arange(m - 1, -1, -1))
    else:
        weights = np.asarray(weights, dtype=float)

    # Default pairs
    if pairs is None:
        pairs = [(i, m) for i in range(m)]

    # Resolve alpha
    if isinstance(alpha, str) and alpha == 'auto':
        alpha = min(max(1000.0 / n, 0.1), 1.0)

    # Compute residuals for each pair
    residuals_x = []
    residuals_y = []
    for x_bit, y_bit in pairs:
        if x_bit == 0:
            res_x = signal.copy()
        else:
            res_x = signal - bits[:, :x_bit] @ weights[:x_bit]

        if y_bit == 0:
            res_y = signal.copy()
        else:
            res_y = signal - bits[:, :y_bit] @ weights[:y_bit]

        residuals_x.append(res_x)
        residuals_y.append(res_y)

    if create_plot:
        n_pairs = len(pairs)
        n_cols = min(n_pairs, 4)
        n_rows = int(np.ceil(n_pairs / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows),
                                 squeeze=False)
        for k, (x_bit, y_bit) in enumerate(pairs):
            row, col = divmod(k, n_cols)
            ax = axes[row][col]
            ax.scatter(residuals_x[k], residuals_y[k], s=4, c='k',
                       alpha=alpha, linewidths=0)
            ax.grid(True)
            ax.set_xlabel('Signal' if x_bit == 0 else f'Res. of bit #{x_bit}')
            ax.set_ylabel('Signal' if y_bit == 0 else f'Res. of bit #{y_bit}')

        # Hide unused subplots
        for k in range(n_pairs, n_rows * n_cols):
            row, col = divmod(k, n_cols)
            axes[row][col].set_visible(False)

        plt.tight_layout()

    return {
        'pairs': pairs,
        'residuals_x': residuals_x,
        'residuals_y': residuals_y,
    }
