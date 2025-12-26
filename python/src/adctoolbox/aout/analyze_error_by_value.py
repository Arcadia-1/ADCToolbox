"""Wrapper for value-based error analysis."""

from typing import Any
import numpy as np
from adctoolbox.aout.rearrange_error_by_value import rearrange_error_by_value
from adctoolbox.aout.plot_rearranged_error_by_value import plot_rearranged_error_by_value

def analyze_error_by_value(
    signal: np.ndarray,
    norm_freq: float = None,
    n_bins: int = 100,
    clip_percent: float = 0.01,
    value_range: tuple[float, float | None] = None,
    create_plot: bool = True,
    axes=None, ax=None,
    title: str = None
) -> dict[str, Any]:
    """
    Analyze error binned by value (INL/DNL/Noise).

    Combines core computation and optional plotting.

    Parameters
    ----------
    signal : np.ndarray
        Input signal (1D array).
    norm_freq : float, optional
        Normalized frequency (f/fs). If None, auto-detected.
    n_bins : int, default=100
        Number of bins for analysis (x-axis resolution).
    clip_percent : float, default=0.01
        Ratio of values to clip from edges.
    value_range : tuple(min, max), optional
        Physical range mapping to bin 0 and bin (N-1).
    create_plot : bool, default=True
        Whether to display result plot.
    axes : tuple or array, optional
        Tuple of (ax1, ax2) to plot on.
    ax : matplotlib.axes.Axes, optional
        Single axis to plot on (will be split).
    title : str, optional
        Test setup description for title.

    Returns
    -------
    results : dict
        Dictionary containing 'error_mean', 'error_rms', 'bin_centers', etc.
    """

    # 1. Compute
    results = rearrange_error_by_value(
        signal=signal,
        norm_freq=norm_freq,
        n_bins=n_bins,
        clip_percent=clip_percent,
        value_range=value_range
    )

    # 2. Plot
    if create_plot:
        plot_rearranged_error_by_value(results, axes=axes, ax=ax, title=title)

    return results