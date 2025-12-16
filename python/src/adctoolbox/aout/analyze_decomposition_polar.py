"""Analyze harmonic decomposition with polar visualization (LMS mode).

This module provides a high-level wrapper that combines LMS harmonic
decomposition calculation with polar plot visualization.

Part of the modular ADC analysis architecture.
Matches MATLAB plotphase.m LMS mode functionality.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any

from .compute_harmonic_decomposition import compute_harmonic_decomposition
from .plot_decomposition_polar import plot_decomposition_polar


def analyze_decomposition_polar(
    data: np.ndarray,
    max_code: Optional[float] = None,
    harmonic: int = 5,
    fs: float = 1.0,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
    ax: Optional[plt.Axes] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Analyze harmonic decomposition with polar visualization (LMS mode).

    This wrapper function combines harmonic decomposition calculation
    with polar plot visualization, following the modular architecture pattern.
    Matches MATLAB plotphase.m LMS mode behavior.

    Parameters
    ----------
    data : np.ndarray
        Input ADC data, shape (N,) for single run or (M, N) for M runs
    max_code : float, optional
        Maximum code level for normalization. If None, uses (max - min)
    harmonic : int, optional
        Number of harmonics to extract and display (default: 5)
    fs : float, optional
        Sampling frequency in Hz (default: 1.0)
    title : str, optional
        Custom title for the plot
    save_path : str or Path, optional
        Path to save the figure. If None, figure is not saved
    show_plot : bool, optional
        Whether to create and display the plot (default: True)
    ax : matplotlib.axes.Axes, optional
        Pre-existing polar axes to plot on. If None, creates new figure

    Returns
    -------
    decomp_result : dict
        Dictionary containing decomposition results from compute_harmonic_decomposition():
        - 'harm_mag', 'harm_phase', 'harm_dB'
        - 'noise_power', 'noise_dB'
        - 'fundamental_freq'
        - 'residual', 'signal_reconstructed'
        - 'fundamental_signal', 'harmonic_signal'
        - 'n_samples', 'fs'

    plot_data : dict
        Dictionary containing data used for plotting

    Examples
    --------
    >>> import numpy as np
    >>> from adctoolbox.aout import analyze_decomposition_polar
    >>>
    >>> # Generate test signal with harmonics
    >>> N = 1000
    >>> t = np.arange(N)
    >>> signal = 0.5 * np.sin(2*np.pi*0.1*t) + 0.05 * np.sin(2*2*np.pi*0.1*t)
    >>>
    >>> # Analyze with polar plot (LMS mode)
    >>> decomp_result, plot_data = analyze_decomposition_polar(
    ...     signal, harmonic=5, fs=1e6, save_path='polar_lms.png'
    ... )
    >>>
    >>> print(f"HD2: {decomp_result['harm_dB'][1]:.1f} dB")
    >>> print(f"Noise floor: {decomp_result['noise_dB']:.1f} dB")

    Notes
    -----
    Modular architecture:
    1. compute_harmonic_decomposition() - Pure calculation (LMS fitting)
    2. plot_decomposition_polar() - Pure visualization (polar plot with noise circle)
    3. analyze_decomposition_polar() - Wrapper combining both

    LMS Mode Features:
    - Uses least-squares fitting to extract harmonics
    - Displays noise circle showing residual error level
    - Harmonics outside noise circle indicate significant distortion
    - Fundamental shown as filled blue circle at phase 0
    - Harmonics shown as hollow blue squares
    """

    # Step 1: Compute harmonic decomposition (pure computation)
    decomp_result = compute_harmonic_decomposition(
        data=data,
        max_code=max_code,
        harmonic=harmonic,
        fs=fs
    )

    # Step 2: Prepare plot data
    plot_data = {
        'harm_mag': decomp_result['harm_mag'],
        'harm_phase': decomp_result['harm_phase'],
        'harm_dB': decomp_result['harm_dB'],
        'noise_dB': decomp_result['noise_dB'],
        'harmonic': harmonic,
        'title': title if title else 'Signal Component Phase (LMS)',
    }

    # Step 3: Plot if requested (pure visualization)
    if show_plot or save_path:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        else:
            fig = ax.get_figure()
            # Verify axes has polar projection
            if not hasattr(ax, 'set_theta_zero_location'):
                raise ValueError("Axes must have polar projection")

        plot_decomposition_polar(plot_data, ax=ax)

        # Save figure if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        # Show plot if requested
        if show_plot and ax is None:
            plt.show()

    return decomp_result, plot_data
