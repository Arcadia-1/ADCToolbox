"""Wrapper for phase-based error analysis (AM/PM decomposition)."""

from typing import Any
import numpy as np
from adctoolbox.aout.rearrange_error_by_phase import rearrange_error_by_phase
from adctoolbox.aout.plot_rearranged_error_by_phase import plot_rearranged_error_by_phase

def analyze_error_by_phase(
    signal: np.ndarray,
    norm_freq: float = None,
    n_bins: int = 100,
    include_base_noise: bool = True,
    create_plot: bool = True,
    axes=None, ax=None,
    title: str = None
) -> dict[str, Any]:
    """Analyze phase error using AM/PM decomposition.

    Uses dual-track parallel design:
    - Path A (Raw): Fit all N samples → highest precision AM/PM values + r_squared_raw
    - Path B (Binned): Compute binned statistics → visualization
    - Cross-validation: Path A coeffs predict Path B trend → r_squared_binned

    Parameters
    ----------
    signal : np.ndarray
        Input signal (1D array).
    norm_freq : float, optional
        Normalized frequency (f/fs), range (0, 0.5). If None, auto-detected via FFT.
    n_bins : int, default=100
        Number of phase bins for visualization.
    include_base_noise : bool, default=True
        Include base noise term in fitting model.
    create_plot : bool, default=True
        Whether to display result plot.
    axes : tuple, optional
        Tuple of (ax1, ax2) for top and bottom panels.
    ax : matplotlib.axes.Axes, optional
        Single axis to split into 2 panels.
    title : str, optional
        Test setup description for title.

    Returns
    -------
    dict
        Numerics: am_noise_rms_v, pm_noise_rms_v, pm_noise_rms_rad, base_noise_rms_v, total_rms_v
        Validation: r_squared_raw (energy ratio), r_squared_binned (model confidence)
        Visualization: bin_error_rms_v, bin_error_mean_v, phase_bin_centers_rad
        Metadata: amplitude, dc_offset, norm_freq, fitted_signal, error, phase
    """
    # 1. Compute
    results = rearrange_error_by_phase(
        signal=signal,
        norm_freq=norm_freq,
        n_bins=n_bins,
        include_base_noise=include_base_noise
    )

    # 2. Plot (always uses binned bar plot)
    if create_plot:
        plot_rearranged_error_by_phase(results, axes=axes, ax=ax, title=title)

    return results
