"""Wrapper for phase-based error analysis."""

from typing import Dict, Any
import numpy as np
from adctoolbox.aout.rearrange_error_by_phase import rearrange_error_by_phase
from adctoolbox.aout.plot_rearranged_error_by_phase import plot_rearranged_error_by_phase


def analyze_error_by_phase(
    signal: np.ndarray,
    normalized_freq: float = None,
    data_mode: str = "binned",
    include_baseline: bool = True,
    bin_count: int = 100,
    show_plot: bool = True,
    axes = None,
    ax = None
) -> Dict[str, Any]:
    """
    Analyze phase error using raw or binned approach (AM/PM decomposition).

    Combines core computation and optional plotting. The plotting visualization
    will automatically match the selected analysis mode.

    Parameters
    ----------
    signal : np.ndarray
        Input signal (1D array).
    normalized_freq : float, optional
        Normalized frequency (f/fs). If None, auto-detected.
    data_mode : str, default="binned"
        Analysis and plotting mode:
        - "raw": Fits to all samples (High precision). Plot shows raw scatter.
        - "binned": Fits to binned RMS (Robust trend). Plot shows binned bars.
    include_baseline : bool, default=True
        Whether to include the baseline noise term in the AM/PM fitting model.
    bin_count : int, default=100
        Number of phase bins (only used if data_mode="binned").
    show_plot : bool, default=True
        Whether to display result plot.
    axes : tuple or array, optional
        Tuple of (ax1, ax2) to plot on.
    ax : matplotlib.axes.Axes, optional
        Single axis to plot on (will be split).

    Returns
    -------
    results : dict
        Dictionary containing analysis results ('am_param', 'pm_param', etc.).
        Structure depends on the selected 'data_mode'.
    """

    # 1. Compute
    results = rearrange_error_by_phase(
        signal=signal,
        normalized_freq=normalized_freq,
        mode=data_mode,  # Pass 'data_mode' to the underlying function's 'mode' argument
        include_baseline=include_baseline,
        bin_count=bin_count
    )

    # 2. Plot (Visualization strictly follows the computation mode)
    if show_plot:
        plot_rearranged_error_by_phase(results, plot_mode=data_mode, axes=axes, ax=ax)

    return results