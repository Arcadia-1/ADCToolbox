"""
Two-tone spectrum analysis for intermodulation distortion (IMD).

Wrapper function combining calculation and plotting following the modular pattern.
Measures IMD2, IMD3 and other spectral metrics with two-tone input.

MATLAB counterpart: specPlot2Tone.m
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union

from .calculate_two_tone_spectrum_data import calculate_two_tone_spectrum_data
from .plot_two_tone_spectrum import plot_two_tone_spectrum


def analyze_two_tone_spectrum(
    data: np.ndarray,
    fs: float = 1.0,
    max_scale_range: Optional[float] = None,
    harmonic: int = 7,
    win_type: str = 'hann',
    side_bin: int = 1,
    show_plot: bool = True,
    show_metrics: bool = True,
    show_labels: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    ax: Optional[plt.Axes] = None
) -> dict:
    """
    Two-tone spectrum analysis with IMD calculation.

    Wrapper function combining calculation and optional plotting.
    Follows the modular architecture pattern.

    Parameters
    ----------
    data : np.ndarray
        ADC output data, shape (M, N) for M runs or (N,) for single run
    fs : float, optional
        Sampling frequency (Hz), default: 1.0
    max_scale_range : float, optional
        Maximum code range, default: max-min of data
    harmonic : int, optional
        Number of harmonics to mark (default: 7)
    win_type : str, optional
        Window type: 'hann', 'blackman', 'hamming', 'boxcar', default: 'hann'
    side_bin : int, optional
        Side bins to include in signal power (default: 1)
    show_plot : bool, optional
        Whether to create plot (default: True)
    show_metrics : bool, optional
        Whether to show metrics on plot (default: True)
    show_labels : bool, optional
        Whether to show frequency/power labels and annotations (default: True)
    save_path : str or Path, optional
        Path to save figure (optional)
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes to plot on. If None and show_plot=True, creates new figure

    Returns
    -------
    dict
        Dictionary with performance metrics:
        - enob: Effective number of bits
        - sndr_db: Signal to noise and distortion ratio (dB)
        - sfdr_db: Spurious free dynamic range (dB)
        - snr_db: Signal to noise ratio (dB)
        - thd_db: Total harmonic distortion (dB)
        - signal_power_1_dbfs: Power of first tone (dBFS)
        - signal_power_2_dbfs: Power of second tone (dBFS)
        - noise_floor_db: Noise floor (dB)
        - imd2_db: 2nd order intermodulation distortion (dB)
        - imd3_db: 3rd order intermodulation distortion (dB)

    Notes
    -----
    Modular architecture:
    1. calculate_two_tone_spectrum_data() - Pure calculation
    2. plot_two_tone_spectrum() - Pure visualization
    3. analyze_two_tone_spectrum() - Wrapper combining both

    Examples
    --------
    >>> import numpy as np
    >>> from adctoolbox import analyze_two_tone_spectrum
    >>>
    >>> # Generate two-tone test signal
    >>> N = 4096
    >>> fs = 1e6
    >>> f1, f2 = 100e3, 120e3
    >>> t = np.arange(N) / fs
    >>> signal = 0.4 * np.sin(2*np.pi*f1*t) + 0.4 * np.sin(2*np.pi*f2*t)
    >>> signal += 0.01 * signal**2  # Add nonlinearity
    >>>
    >>> # Analyze
    >>> metrics = analyze_two_tone_spectrum(signal, fs=fs)
    >>> print(f"IMD2: {metrics['imd2_db']:.2f} dB")
    >>> print(f"IMD3: {metrics['imd3_db']:.2f} dB")
    """

    # Step 1: Calculate spectrum data (pure computation)
    results = calculate_two_tone_spectrum_data(
        data=data,
        fs=fs,
        max_scale_range=max_scale_range,
        win_type=win_type,
        side_bin=side_bin
    )

    # Step 2: Plot if requested (pure visualization)
    if show_plot or save_path:
        if ax is None and show_plot:
            fig, ax = plt.subplots(figsize=(10, 6))

        plot_two_tone_spectrum(
            analysis_results=results,
            harmonic=harmonic,
            ax=ax,
            show_metrics=show_metrics,
            show_labels=show_labels
        )

        # Save figure if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        # Show plot if requested and no external axes provided
        if show_plot and ax is None:
            plt.show()

        # Close figure if created internally and not showing
        if not show_plot and ax is None:
            plt.close()

    # Step 3: Return metrics dictionary (like analyze_spectrum.py)
    return results['metrics']

