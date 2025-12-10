"""Analyze harmonic decomposition with time-domain visualization.

This module provides a high-level wrapper that combines LMS harmonic
decomposition calculation with time-domain visualization.

Part of the modular ADC analysis architecture.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any

from .calculate_lms_decomposition import calculate_lms_decomposition
from .plot_decomposition_time import plot_decomposition_time


def analyze_decomposition_time(
    data: np.ndarray,
    max_code: Optional[float] = None,
    harmonic: int = 5,
    fs: float = 1.0,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
    ax: Optional[plt.Axes] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Analyze harmonic decomposition with time-domain visualization.

    This wrapper function combines LMS harmonic decomposition calculation
    with time-domain plotting, following the modular architecture pattern.

    Parameters
    ----------
    data : np.ndarray
        Input ADC data, shape (N,) for single run or (M, N) for M runs
    max_code : float, optional
        Maximum code level for normalization. If None, uses (max - min)
    harmonic : int, optional
        Number of harmonics to extract (default: 5)
    fs : float, optional
        Sampling frequency in Hz (default: 1.0)
    title : str, optional
        Custom title for the plot
    save_path : str or Path, optional
        Path to save the figure. If None, figure is not saved
    show_plot : bool, optional
        Whether to create and display the plot (default: True)
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes to plot on. If None, creates new figure

    Returns
    -------
    decomp_result : dict
        Dictionary containing decomposition results from calculate_lms_decomposition():
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
    >>> from adctoolbox.aout import analyze_decomposition_time
    >>>
    >>> # Generate test signal
    >>> N = 1000
    >>> t = np.arange(N)
    >>> signal = 0.5 * np.sin(2*np.pi*0.1*t) + 0.01*np.random.randn(N)
    >>>
    >>> # Analyze with time-domain plot
    >>> decomp_result, plot_data = analyze_decomposition_time(
    ...     signal, harmonic=5, fs=1e6, save_path='decomp.png'
    ... )
    >>>
    >>> print(f"Fundamental frequency: {decomp_result['fundamental_freq']:.6f}")
    >>> print(f"Noise floor: {decomp_result['noise_dB']:.1f} dB")

    Notes
    -----
    Modular architecture:
    1. calculate_lms_decomposition() - Pure calculation
    2. plot_decomposition_time() - Pure visualization
    3. analyze_decomposition_time() - Wrapper combining both
    """

    # Step 1: Calculate LMS harmonic decomposition (pure computation)
    decomp_result = calculate_lms_decomposition(
        data=data,
        max_code=max_code,
        harmonic=harmonic,
        fs=fs
    )

    # Step 2: Prepare plot data
    # Get original signal (denormalized)
    data = np.asarray(data)
    if data.ndim == 1:
        signal = data
    elif data.ndim == 2:
        signal = np.mean(data, axis=0)
    else:
        raise ValueError(f"Input data must be 1D or 2D, got {data.ndim}D")

    # Denormalize signals for plotting
    if max_code is None:
        max_code = np.max(signal) - np.min(signal)

    signal_mean = np.mean(signal)

    plot_data = {
        'signal': signal - signal_mean,  # DC removed for consistency
        'fundamental_signal': decomp_result['fundamental_signal'] * max_code,
        'harmonic_signal': decomp_result['harmonic_signal'] * max_code,
        'residual': decomp_result['residual'] * max_code,
        'fundamental_freq': decomp_result['fundamental_freq'],
        'title': title if title else 'Harmonic Decomposition (Time Domain)',
    }

    # Step 3: Plot if requested (pure visualization)
    if show_plot or save_path:
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig = ax.get_figure()

        plot_decomposition_time(plot_data, ax=ax)

        # Save figure if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        # Show plot if requested
        if show_plot and ax is None:
            plt.show()

    return decomp_result, plot_data
