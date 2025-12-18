"""Wrapper for harmonic decomposition analysis with polar visualization."""

from typing import Optional, Dict, Any
import numpy as np
from adctoolbox.aout.decompose_harmonic_error import decompose_harmonic_error
from adctoolbox.aout.plot_decomposition_polar import plot_decomposition_polar


def analyze_decomposition_polar(
    signal: np.ndarray,
    harmonic: int = 5,
    show_plot: bool = True,
    ax: Optional[object] = None,
    title: str = None
) -> Dict[str, Any]:
    """
    Analyze harmonic decomposition with polar visualization.

    Combines core computation and optional plotting.

    Parameters
    ----------
    signal : np.ndarray
        Input signal (1D array).
    harmonic : int, default=5
        Number of harmonics to extract.
    show_plot : bool, default=True
        Whether to display result plot.
    ax : matplotlib.axes.Axes, optional
        Polar axis to plot on.
    title : str, optional
        Custom title for the plot.

    Returns
    -------
    results : dict
        Dictionary containing decomposition results from decompose_harmonic_error().
    """

    # 1. Compute
    results = decompose_harmonic_error(
        signal=signal,
        n_harmonics=harmonic
    )

    # 2. Plot
    if show_plot:
        plot_decomposition_polar(
            results=results,
            harmonic=harmonic,
            ax=ax,
            title=title
        )

    return results
