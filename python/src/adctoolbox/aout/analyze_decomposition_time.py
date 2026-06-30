"""Wrapper for harmonic decomposition analysis with time-domain visualization."""

from typing import Any
import numpy as np
from adctoolbox.aout.decompose_harmonic_error import decompose_harmonic_error
from adctoolbox.aout.plot_decomposition_time import plot_decomposition_time

def analyze_decomposition_time(
    signal: np.ndarray,
    harmonic: int = 5,
    n_cycles: float = 5.0,
    create_plot: bool = True,
    ax=None,
    title: str = None,
    frequency: float = None,
    max_iterations: int = 1,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """
    Analyze harmonic decomposition with time-domain visualization.

    Combines core computation and optional plotting.

    Parameters
    ----------
    signal : np.ndarray
        Input signal (1D array).
    harmonic : int, default=5
        Number of harmonics to extract.
    n_cycles : float, default=5.0
        Number of cycles to display in the time-domain plot.
    create_plot : bool, default=True
        Whether to display result plot.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on (will be split for multi-panel).
    title : str, optional
        Custom title for the plot.
    frequency : float, optional
        Normalized fundamental frequency (0 to 0.5). If None, auto-detected.
    max_iterations : int, default=1
        Frequency-refinement iterations passed to fit_sine_4param.
    tolerance : float, default=1e-9
        Frequency-refinement convergence threshold passed to fit_sine_4param.

    Returns
    -------
    results : dict
        Dictionary containing decomposition results from decompose_harmonic_error().
    """

    # 1. Compute
    results = decompose_harmonic_error(
        signal=signal,
        n_harmonics=harmonic,
        frequency=frequency,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )

    # 2. Plot
    if create_plot:
        plot_decomposition_time(
            results=results,
            signal=signal,
            n_cycles=n_cycles,
            ax=ax,
            title=title
        )

    return results

