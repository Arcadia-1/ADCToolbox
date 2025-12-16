"""Wrapper for harmonic decomposition analysis with time-domain visualization."""

from typing import Optional, Dict, Any
import numpy as np
from .compute_harmonic_decomposition import compute_harmonic_decomposition
from .plot_decomposition_time import plot_decomposition_time


def analyze_decomposition_time(
    signal: np.ndarray,
    harmonic: int = 5,
    fs: float = 1.0,
    show_plot: bool = True,
    ax: Optional[object] = None
) -> Dict[str, Any]:
    """
    Analyze harmonic decomposition with time-domain visualization.

    Combines core computation and optional plotting.

    Parameters
    ----------
    signal : np.ndarray
        Input signal (1D array).
    harmonic : int, default=5
        Number of harmonics to extract.
    fs : float, default=1.0
        Sampling frequency.
    show_plot : bool, default=True
        Whether to display result plot.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on (will be split for multi-panel).

    Returns
    -------
    results : dict
        Dictionary containing decomposition results from compute_harmonic_decomposition().
    """

    # 1. Compute
    results = compute_harmonic_decomposition(
        data=signal,
        max_code=None,
        harmonic=harmonic,
        fs=fs
    )

    # 2. Plot
    if show_plot:
        # Prepare plot_data with required keys
        plot_data = results.copy()
        plot_data['signal'] = signal
        plot_data['fundamental_freq'] = results.get('fundamental_freq', 0.1)
        plot_decomposition_time(plot_data, ax=ax)

    return results

